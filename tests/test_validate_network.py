"""Tests for the validation gate.

The fast tests build tiny ONNX graphs by hand, so they need neither HSSM nor a
network. The end-to-end test against the real production ddm.onnx is opt-in
(`-m production`): it downloads from HuggingFace and imports the whole
inference stack, which does not belong in the default suite.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import validate_network as vn
from validate_network import (
    GONOGO_SKIP_REASON,
    gate_accuracy,
    gate_parity,
    gate_structure,
    hellinger,
    validate_network,
)


def make_onnx(path, input_dims, output_dims=(1, 1), real_out_width=None):
    """A minimal MatMul graph with the requested input and output dims.

    dims entries may be ints (concrete) or strings (symbolic), which is the
    distinction G1 exists to enforce. The weight is sized from both, so an
    output width other than 1 produces a *valid* graph that G1 must reject on
    the contract rather than on the checker.

    ``real_out_width`` sizes the weight independently of the declared output
    shape, which is how a graph whose annotation and behaviour disagree gets
    built — the case that decides whether the gate reads metadata or measures.
    """
    in_width = input_dims[-1] if isinstance(input_dims[-1], int) else 6
    out_width = real_out_width or (
        output_dims[-1] if isinstance(output_dims[-1], int) else 1
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_dims))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(output_dims))
    weight = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [in_width, out_width],
        np.zeros((in_width, out_width), dtype=np.float32).ravel().tolist(),
    )
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def make_constant_onnx(path, width, value):
    """A bias-only graph: zero weights, so every input maps to ``value``.

    The shape of an auxiliary network that has learned nothing, and the
    cheapest way to hand gate_accuracy a known output.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [width, 1], np.zeros(width, dtype=np.float32).tolist()
    )
    bias = helper.make_tensor("b", TensorProto.FLOAT, [1, 1], [float(value)])
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"]),
        helper.make_node("Add", ["h", "b"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def make_two_output_onnx(path, input_dims=(1, 6)):
    """A valid graph that emits two separate 1-wide tensors."""
    width = input_dims[-1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_dims))
    outs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1]) for n in ("y", "z")
    ]
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [width, 1], np.zeros(width, dtype=np.float32).tolist()
    )
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["y"]),
        helper.make_node("Identity", ["y"], ["z"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], outs, initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


class TestStructureGate:
    def test_accepts_a_concrete_single_trial_graph(self, tmp_path):
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        result = gate_structure(path, expected_input_dim=6)
        assert result["passed"], result
        assert result["input_width"] == 6

    def test_rejects_a_symbolic_batch_dim(self, tmp_path):
        # HSSM's make_jax_func raises on symbolic dims, so this would fail at
        # load for every user rather than here.
        path = make_onnx(tmp_path / "dyn.onnx", ("batch", 6))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "symbolic" in result["error"]

    def test_rejects_a_width_that_contradicts_the_parameter_space(self, tmp_path):
        # ddm has 4 params, so a LAN must take 4 + rt + response = 6.
        path = make_onnx(tmp_path / "narrow.onnx", (1, 5))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "expected 6" in result["error"]

    def test_rejects_an_output_wider_than_one_log_density(self, tmp_path):
        # A per-choice output head is a valid ONNX graph and the wrong
        # likelihood; G2 and G4 would score its first column and say nothing.
        path = make_onnx(tmp_path / "wide.onnx", (1, 6), output_dims=(1, 2))
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "2 values returned for one trial" in result["error"]

    def test_a_symbolic_output_annotation_is_judged_on_what_it_returns(self, tmp_path):
        # HSSM never reads the declared output shape (onnx2jax validates
        # graph.input dims and resolves outputs by name), so a symbolic
        # annotation over a genuinely 1-wide output is cosmetic. The gate runs
        # the graph instead of trusting either annotation.
        good = make_onnx(tmp_path / "sym_ok.onnx", (1, 6), output_dims=(1, "d"))
        assert gate_structure(good, expected_input_dim=6)["passed"]

        # The case metadata cannot catch: annotated 1 wide, actually 2 wide.
        lying = make_onnx(
            tmp_path / "lying.onnx", (1, 6), output_dims=(1, 1), real_out_width=2
        )
        result = gate_structure(lying, expected_input_dim=6)
        assert not result["passed"]
        assert "2 values returned for one trial" in result["error"]

    def test_rejects_a_graph_with_more_than_one_output(self, tmp_path):
        path = make_two_output_onnx(tmp_path / "two.onnx")
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "exactly 1 input and 1 output" in result["error"]

    def test_reports_a_corrupt_file_rather_than_raising(self, tmp_path):
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"not an onnx file")
        result = gate_structure(path, expected_input_dim=6)
        assert not result["passed"]
        assert "error" in result


class TestParityGate:
    def test_skips_without_a_flax_state(self, tmp_path):
        # Torch-trained and downloaded networks have no .jax sibling; that is
        # a skip, not a failure.
        result = gate_parity(tmp_path / "m.onnx", None, None, input_width=6)
        assert result["passed"] and result["skipped"]


class TestHellinger:
    def test_identical_distributions_are_zero(self):
        assert hellinger([1, 2, 3], [1, 2, 3]) == 0.0

    def test_disjoint_distributions_are_one(self):
        assert hellinger([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_is_scale_invariant(self):
        # Inputs are normalized, so unnormalized histograms compare correctly.
        assert hellinger([1, 1], [50, 50]) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a, b = [0.7, 0.2, 0.1], [0.3, 0.4, 0.3]
        assert hellinger(a, b) == pytest.approx(hellinger(b, a))


class TestWiring:
    def test_a_broken_graph_short_circuits_the_expensive_gates(self, tmp_path):
        """G3/G4 load HSSM and run simulations; there is nothing to learn from
        running them once the graph itself is unusable."""
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"nope")
        report = validate_network(path, model_name="ddm", network_type="lan")
        assert not report["passed"]
        gates = {g["gate"]: g for g in report["gates"]}
        assert not gates["structure"]["passed"]
        for later in ("parity", "hssm_load", "density"):
            assert gates[later].get("skipped"), later

    def test_an_unknown_network_type_is_rejected_not_defaulted(self, tmp_path):
        # Defaulting to 0 extra inputs would surface as "input width 6 !=
        # expected 4", blaming the artifact for a mistyped flag.
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        with pytest.raises(ValueError, match="Unknown network_type"):
            validate_network(path, model_name="ddm", network_type="LAN")

    def test_report_shape_is_stable(self, tmp_path):
        path = make_onnx(tmp_path / "good.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", skip_hssm=True, skip_density=True
        )
        assert report["schema_version"] == 1
        assert [g["gate"] for g in report["gates"]] == [
            "structure",
            "parity",
            "hssm_load",
            "density",
        ]
        # The one top-level key the auxiliary gates added; null for a LAN.
        assert report["aux_category"] is None


AUX_SKIPS = dict(skip_hssm=True, skip_accuracy=True)


class TestAuxiliaryWidths:
    """ddm has 4 params: an auxiliary net takes them plus one trailing input."""

    @pytest.mark.parametrize(
        "network_type, aux_category",
        [("cpn", "choice"), ("opn", None), ("gonogo", None)],
    )
    def test_n_params_plus_one_is_accepted(self, tmp_path, network_type, aux_category):
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        assert report["gates"][0]["passed"], report["gates"][0]

    @pytest.mark.parametrize(
        "network_type, aux_category", [("cpn", "choice"), ("opn", None)]
    )
    def test_the_old_params_only_width_is_rejected(
        self, tmp_path, network_type, aux_category
    ):
        # The pre-contract CPN/OPN took the parameters alone. HSSM's fixture
        # ddm_cpn.onnx is still that shape, and this is what refuses it.
        path = make_onnx(tmp_path / "old.onnx", (1, 4))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        structure = report["gates"][0]
        assert not structure["passed"]
        assert "expected 5" in structure["error"]

    def test_a_lan_still_takes_rt_and_response(self, tmp_path):
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        report = validate_network(
            path, model_name="ddm", skip_hssm=True, skip_density=True
        )
        assert report["gates"][0]["passed"]


class TestAuxiliaryWiring:
    @pytest.mark.parametrize(
        "network_type, aux_category", [("cpn", "choice"), ("opn", None)]
    )
    def test_an_aux_report_lists_exactly_the_aux_gates(
        self, tmp_path, network_type, aux_category
    ):
        path = make_onnx(tmp_path / "aux.onnx", (1, 5))
        report = validate_network(
            path,
            model_name="ddm",
            network_type=network_type,
            aux_category=aux_category,
            **AUX_SKIPS,
        )
        assert [g["gate"] for g in report["gates"]] == [
            "structure",
            "parity",
            "hssm_missing_load",
            "accuracy",
        ]
        assert report["network_type"] == network_type

    def test_opn_defaults_its_category_to_deadline(self, tmp_path):
        path = make_onnx(tmp_path / "opn.onnx", (1, 5))
        report = validate_network(
            path, model_name="ddm", network_type="opn", **AUX_SKIPS
        )
        assert report["aux_category"] == "deadline"

    def test_gonogo_reports_its_two_unrunnable_gates_as_skipped(self, tmp_path):
        path = make_onnx(tmp_path / "gonogo.onnx", (1, 5))
        report = validate_network(path, model_name="ddm", network_type="gonogo")
        gates = {g["gate"]: g for g in report["gates"]}
        for name in ("hssm_missing_load", "accuracy"):
            assert gates[name]["skipped"], name
            assert gates[name]["reason"] == GONOGO_SKIP_REASON

    def test_a_deadline_model_name_is_rejected(self, tmp_path):
        # The deadline variant is derived where it is needed; naming it would
        # double the deadline in every simulation.
        path = make_onnx(tmp_path / "opn.onnx", (1, 5))
        with pytest.raises(ValueError, match="base model"):
            validate_network(
                path, model_name="ddm_deadline", network_type="opn", **AUX_SKIPS
            )

    def test_cpn_without_a_category_fails_naming_the_flag(self, tmp_path):
        path = make_onnx(tmp_path / "cpn.onnx", (1, 5))
        with pytest.raises(ValueError, match="--aux-category"):
            validate_network(path, model_name="ddm", network_type="cpn", **AUX_SKIPS)

    def test_a_category_on_a_lan_is_rejected(self, tmp_path):
        path = make_onnx(tmp_path / "lan.onnx", (1, 6))
        with pytest.raises(ValueError, match="--aux-category"):
            validate_network(
                path,
                model_name="ddm",
                aux_category="choice",
                skip_hssm=True,
                skip_density=True,
            )

    def test_a_broken_aux_graph_short_circuits_the_aux_gates(self, tmp_path):
        path = tmp_path / "junk.onnx"
        path.write_bytes(b"nope")
        report = validate_network(path, model_name="ddm", network_type="opn")
        gates = {g["gate"]: g for g in report["gates"]}
        for later in ("parity", "hssm_missing_load", "accuracy"):
            assert gates[later].get("skipped"), later

    def test_cpn_missing_load_waits_for_an_hssm_that_passes_response(
        self, tmp_path, monkeypatch
    ):
        # Under the locked hssm 0.4.0 HSSM calls a CPN with the parameters
        # only, so a contract-conformant width n_params + 1 CPN cannot load.
        # The gate must say so as a skip, and flip on by itself once the lock
        # moves past 0.6.0 — hence the version is patched both ways here.
        path = make_onnx(tmp_path / "cpn.onnx", (1, 5))
        monkeypatch.setattr(
            vn, "_hssm_version_supports_cpn_response", lambda: (False, "0.4.0")
        )
        report = validate_network(
            path,
            model_name="ddm",
            network_type="cpn",
            aux_category="choice",
            skip_accuracy=True,
        )
        gate = {g["gate"]: g for g in report["gates"]}["hssm_missing_load"]
        assert gate["skipped"] and gate["passed"]
        assert gate["reason"] == "HSSM < 0.6.0 ignores response on missing rows"
        assert gate["hssm_version"] == "0.4.0"

    def test_the_real_version_check_reads_the_installed_hssm(self):
        from importlib.metadata import version

        from packaging.version import Version

        supported, installed = vn._hssm_version_supports_cpn_response()
        assert installed == version("hssm")
        assert supported == (Version(installed) >= Version("0.6.0"))


class TestAccuracyGate:
    """gate_accuracy against a bias-only graph and a monkeypatched truth.

    cpn is used because its truth needs no simulation once patched; opn would
    still simulate the base model to place its deadline.
    """

    @staticmethod
    def constant_truth(p):
        def fake_truth(model_name, network_type, theta, **kwargs):
            assert network_type == "cpn"
            assert "choice" in kwargs
            return {"truth": p, "truth_mc_se": 0.0016, "n_sim": 100_000}

        return fake_truth

    def test_passes_when_the_network_matches_the_truth(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.5))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert result["passed"], result
        assert result["max_abs_error"] == pytest.approx(0.0, abs=1e-6)
        assert len(result["draws"]) == 4
        # Every record carries what a reader needs to tell network error from
        # simulation noise.
        for draw in result["draws"]:
            assert draw["truth_mc_se"] == 0.0016
            assert draw["truth"] == 0.5
            assert draw["network_value"] == pytest.approx(0.5, abs=1e-6)
            assert len(draw["theta"]) == 4
        # The choice code cycles over the model's declared choices.
        assert [d["choice"] for d in result["draws"]] == [-1.0, 1.0, -1.0, 1.0]

    def test_fails_when_the_network_is_far_from_the_truth(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.9))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert not result["passed"]
        assert result["mean_abs_error"] == pytest.approx(0.4, abs=1e-6)
        assert result["mean_abs_max"] == vn.DEFAULT_ACCURACY_MEAN_ABS_MAX
        assert result["max_abs_max"] == vn.DEFAULT_ACCURACY_MAX_ABS_MAX

    def test_thresholds_are_parameters_not_constants(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "half.onnx", 5, np.log(0.5))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.52))
        assert not gate_accuracy(path, "ddm", "cpn", n_param_draws=2)["passed"]
        loose = gate_accuracy(
            path, "ddm", "cpn", n_param_draws=2, mean_abs_max=0.05, max_abs_max=0.05
        )
        assert loose["passed"]

    def test_a_positive_output_is_not_a_log_probability(self, tmp_path, monkeypatch):
        # A raw-logit export (no log-sigmoid in the graph) emits values above
        # zero. That is a contract failure before any truth is worth computing.
        path = make_constant_onnx(tmp_path / "logit.onnx", 5, 0.3)
        called = []
        monkeypatch.setattr(
            vn, "aux_truth", lambda *a, **k: called.append(1) or {"truth": 0.5}
        )
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=4)
        assert not result["passed"]
        assert "finite and <= 0" in result["error"]
        assert called == []

    def test_a_non_finite_output_fails_the_same_way(self, tmp_path, monkeypatch):
        path = make_constant_onnx(tmp_path / "nan.onnx", 5, float("nan"))
        monkeypatch.setattr(vn, "aux_truth", self.constant_truth(0.5))
        result = gate_accuracy(path, "ddm", "cpn", n_param_draws=1)
        assert not result["passed"]
        assert "finite and <= 0" in result["error"]


class TestAuxTruth:
    def test_truth_carries_its_monte_carlo_se(self, monkeypatch):
        # A stand-in simulator: 30% of trials pick +1, none run past max_t.
        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            choices = np.where(np.arange(n_samples) % 10 < 3, 1, -1)
            return np.full(n_samples, 0.5), choices

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "cpn", np.zeros(4), choice=1, n_sim=1000)
        assert truth["truth"] == pytest.approx(0.3)
        assert truth["truth_mc_se"] == pytest.approx(np.sqrt(0.3 * 0.7 / 1000))
        assert truth["truth_rt_lt_max_t"] == pytest.approx(0.3)

    def test_cpn_truth_separates_trials_that_outlast_max_t(self, monkeypatch):
        # ssms hands an un-terminated trial back at rt ≈ max_t + t with a
        # sign-implied choice; the gate's truth keeps it, the windowed one
        # does not, and both are reported.
        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            rts = np.where(np.arange(n_samples) % 2 == 0, 0.5, 20.4)
            return rts, np.ones(n_samples)

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "cpn", np.zeros(4), choice=1, n_sim=100)
        assert truth["truth"] == pytest.approx(1.0)
        assert truth["truth_rt_lt_max_t"] == pytest.approx(0.5)

    def test_opn_truth_is_the_omission_rate_of_the_deadline_variant(self, monkeypatch):
        seen = {}

        def fake_simulate(model_name, theta, n_samples, random_state, max_t=20.0):
            seen["model"], seen["theta"] = model_name, np.asarray(theta)
            rts = np.where(np.arange(n_samples) % 4 == 0, -999.0, 0.7)
            # An omission is marked in rts only; choices stay sign-implied.
            return rts, np.ones(n_samples)

        monkeypatch.setattr(vn, "_simulate", fake_simulate)
        truth = vn.aux_truth("ddm", "opn", np.arange(4.0), deadline=1.5, n_sim=400)
        assert seen["model"] == "ddm_deadline"
        assert seen["theta"].tolist() == [0.0, 1.0, 2.0, 3.0, 1.5]
        assert truth["truth"] == pytest.approx(0.25)
        assert "truth_rt_lt_max_t" not in truth

    def test_gonogo_has_no_truth(self):
        with pytest.raises(ValueError, match="gonogo"):
            vn.aux_truth("ddm", "gonogo", np.zeros(4), deadline=1.0)


@pytest.mark.production
class TestAgainstProduction:
    """The gate's own acceptance test: it must pass a network that works.

    Opt-in (`-m production`): downloads from HuggingFace and imports HSSM.
    """

    def test_production_ddm_passes_every_gate(self):
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        onnx_path = Path(hf_hub_download("franklab/HSSM", "ddm.onnx"))
        report = validate_network(onnx_path, model_name="ddm", network_type="lan")
        gates = {g["gate"]: g for g in report["gates"]}
        assert gates["structure"]["passed"], gates["structure"]
        assert gates["hssm_load"]["passed"], gates["hssm_load"]
        assert gates["density"]["passed"], gates["density"]
        # Comfortably inside the bound, not scraping it.
        assert gates["density"]["worst_ratio"] < 2.5
        assert report["passed"]
