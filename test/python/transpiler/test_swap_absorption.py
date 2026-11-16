# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the SWAP absorption pass: the rewrites, the checks, odd inputs and the presets."""

import itertools
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from ddt import data, ddt, idata, unpack

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CPhaseGate, CRZGate, CU1Gate, CXGate, QFTGate, RZZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManagerConfig, Target, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasisTranslator, CheckGateDirection, HLSConfig
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps, _rewrites
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from test import QiskitTestCase

THETA = 0.7
CZ_BASIS = ["cz", "rz", "sx", "x"]
LEVELS, METHODS = [None, 0, 1, 2, 3], ["default", "translator", "synthesis"]
BASES = [CZ_BASIS, [*CZ_BASIS, "swap"], [*CZ_BASIS, "rzz"]]
# Two-qubit gate weights before translation (SWAP is three CX, a controlled phase two).
WEIGHTS = {"swap": 3, "cx": 1, "cz": 1, "ecr": 1}
# The four variants of a row: wires as written or with 0 and 1 exchanged, forward or reversed.
VARIANTS = list(itertools.product([(0, 1, 2, 3), (1, 0, 2, 3)], [1, -1]))


def two_qubit_weight(qc):
    """The weight of the two-qubit gates of ``qc`` before translation."""
    return sum(WEIGHTS.get(inst.operation.name, 2) for inst in qc.data if len(inst.qubits) == 2)


def circuit(gates, num_qubits=4, qubits=(0, 1, 2, 3)):
    """A circuit from ``(name, qubits, *params)`` rows; ``qubits`` renames the wires."""
    qc = QuantumCircuit(num_qubits)
    for name, wires, *params in gates:
        getattr(qc, name)(*params, *(qubits[wire] for wire in wires))
    return qc


def two_qubit_row(name):
    """A function that makes ``(name, (a, b))`` rows."""
    return lambda a, b: (name, (a, b))


cx, cz, dcx, ecr, swap = map(two_qubit_row, ["cx", "cz", "dcx", "ecr", "swap"])


def line_target(num_qubits, basis=CZ_BASIS):
    """A target with ``basis`` on a line of ``num_qubits`` qubits."""
    return Target.from_configuration(basis, num_qubits, CouplingMap.from_line(num_qubits))


SWAP = swap(0, 1)
RZ = ("rz", (0,), THETA)
BARRIER = ("barrier", (0, 1, 2, 3))

# (gates, operation counts after the rewrite); the test adds a barrier so that no SWAP ends a wire
REWRITES = (
    # Z on the control wire of the later gate fits g = CX with that control
    ((SWAP, cx(0, 2), cx(0, 1)), {"cx": 3}),
    ((SWAP, cx(1, 2), cx(1, 0)), {"cx": 3}),
    ((swap(1, 2), cx(2, 3), cx(2, 1)), {"cx": 3}),
    # Z on wire 0 and nothing on wire 1: g = CX(0, 1)
    ((SWAP, cx(0, 2), SWAP), {"cx": 5}),
    ((SWAP, cx(0, 2), swap(1, 0)), {"cx": 5}),
    # Z on both wires: no CX fits as g, CZ does
    ((SWAP, cx(0, 2), cx(1, 3), SWAP), {"cx": 6, "h": 4}),
    ((SWAP, cx(0, 2), ("rz", (1,), THETA), cz(1, 0)), {"cx": 3, "h": 2, "rz": 1}),
    ((SWAP, cx(0, 2), cx(1, 3), ("cp", (0, 1), THETA)), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), ("cp", (1, 0), THETA)), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), ("crz", (0, 1), THETA)), {"cx": 5, "p": 2}),
    ((SWAP, cx(0, 2), cx(1, 3), ("crz", (1, 0), THETA)), {"cx": 5, "p": 2}),
    ((SWAP, cx(0, 2), cx(1, 3), ("rzz", (0, 1), THETA)), {"cx": 5, "p": 1}),
    ((SWAP, cx(0, 2), cx(1, 3), ("rzz", (1, 0), THETA)), {"cx": 5, "p": 1}),
    ((SWAP, cx(0, 2), cx(1, 3), ("cs", (0, 1))), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), ("cs", (1, 0))), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), ("csdg", (0, 1))), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), ("csdg", (1, 0))), {"cx": 5, "p": 3}),
    ((SWAP, cx(0, 2), cx(1, 3), cx(0, 1), ("rz", (1,), THETA), cx(0, 1)), {"cx": 5, "p": 1}),
    ((SWAP, cx(0, 2), cx(1, 3), cx(1, 0), ("rz", (0,), THETA), cx(1, 0)), {"cx": 5, "p": 1}),
)

# (reason, before, rejected, control, after, move_diagonal): only the control circuit is rewritten
CHECKS = (
    # X on the control wire of the later gate: no g fits
    ("no_clifford", (SWAP,), (cx(2, 0), cx(0, 1)), (cx(0, 2), cx(0, 1)), (BARRIER,)),
    ("no_clifford", (SWAP,), (cx(2, 1), cx(1, 0)), (cx(1, 2), cx(1, 0)), (BARRIER,)),
    ("C1", (SWAP,), (RZ, cx(0, 1)), (cx(0, 2), RZ, cx(0, 1)), ()),
    # the peephole collects the two CX gates into one block
    ("C1", (SWAP, cx(0, 2), cx(0, 1)), (cx(1, 0),), (), ()),
    # cx(0, 2) becomes adjacent to cx(2, 0) when cx(0, 1) is removed
    ("C2", (SWAP, cx(0, 2), cx(0, 1), cx(2, 0), cx(2, 3)), (cx(2, 0),), (cx(0, 2),), ()),
    # g = CX(1, 0) changes the SWAP's far end, where the h on wire 1 may be merged
    ("C3", (), (cx(0, 1),), (cx(1, 0),), (cx(0, 3), ("h", (1,)), SWAP, cx(2, 0), cx(1, 0))),
    ("C3", (SWAP, cx(0, 2), cx(0, 1), cx(0, 3)), (cx(0, 1),), (cx(1, 0),), ()),
    # the ecr after the second SWAP exposes cx(0, 1) to that SWAP's far end through cz(0, 2)
    ("C3", (SWAP, cx(2, 0), cz(1, 3), SWAP, cz(0, 2)), (ecr(0, 1),), (ecr(1, 0),), (BARRIER,)),
    # dcx(0, 1) is cx(0, 1); cx(1, 0): its start exposes cx(0, 1) to a following walk
    ("C3", (SWAP, cx(2, 0), cz(1, 3), SWAP, cx(0, 3)), (dcx(0, 1),), (dcx(1, 0),), (BARRIER,)),
    # its end exposes cx(1, 0) to a preceding walk
    (
        "C3",
        (),
        (dcx(1, 0),),
        (dcx(0, 1),),
        (cx(0, 3), SWAP, cx(2, 0), cz(1, 3), SWAP, cx(0, 3), BARRIER),
    ),
    ("C3", (SWAP, cx(0, 2), ("rz", (1,), THETA), cz(0, 1), cx(0, 3)), (cz(1, 0),), (cx(0, 1),), ()),
    ("C4", (SWAP, cx(0, 2)), (("cp", (0, 1), THETA),), (("cp", (1, 0), THETA),), ()),
    # with ``move_diagonal=False`` a controlled phase is no partner, a CZ is
    ("partner", (SWAP, cx(0, 2), cx(1, 3)), (("cp", (1, 0), THETA),), (cz(1, 0),), (), False),
)

# (gates, stats with ``peephole_in_loop``, rewrites without)
ONE_REWRITE = (
    # a routing ladder step and its return across Z gaps, flanked on wire 1 by a third SWAP
    ((SWAP, cx(0, 2), SWAP, cx(0, 2), SWAP, cx(1, 2)), {"applied": 1}, 1),
    # a fourth SWAP beyond the flank repeats the gate the rewritten SWAP would meet there
    ((SWAP, cx(0, 2), SWAP, cx(0, 2), SWAP, SWAP, cx(1, 2)), {"C1": 4}, 1),
    # a cx(0, 1) there does not
    ((SWAP, cx(0, 2), SWAP, cx(0, 2), SWAP, cx(0, 1), cx(1, 2)), {"applied": 1, "C1": 1}, 1),
    ((SWAP, cx(2, 0), SWAP, cx(2, 0), SWAP, cx(1, 2)), {"applied": 1}, 1),
    ((SWAP, cx(0, 2), SWAP, cx(0, 2), cx(1, 2), cx(1, 2)), {"applied": 1}, 1),
    # the second SWAP is the partner of the first, so the third finds no partner left
    ((SWAP, cx(0, 2), SWAP, cx(0, 2), SWAP, BARRIER), {"applied": 1}, 1),
    # the loop collapses the swaps on wires 1 and 3, then the two cx(0, 2) cancel
    ((cx(0, 2), SWAP, swap(1, 3), swap(1, 3), SWAP, cx(0, 2), swap(1, 0)), {"C1": 2}, 1),
    ((SWAP, swap(1, 3), swap(1, 3), SWAP, cx(0, 2), swap(1, 0)), {"applied": 1, "C1": 1}, 1),
    (
        (cx(2, 0), SWAP, swap(1, 3), swap(1, 3), SWAP, cx(2, 0), swap(1, 0)),
        {"applied": 1, "C1": 1},
        1,
    ),
    # a SWAP with nothing after it on a wire is left alone
    ((cx(0, 1), cx(0, 2), SWAP), {}, 0),
    # two SWAPs whose diagonal partners give the same block
    ((SWAP, swap(2, 3), cz(0, 2), cz(1, 3), ("cs", (0, 1)), ("cs", (2, 3))), {"applied": 2}, 2),
)

ONE_WAY = Target.from_configuration(["cx"], 4, CouplingMap([(0, 1), (0, 2), (2, 0)]))
TWO_WAY = Target.from_configuration(["cx"], 4, CouplingMap([(0, 1), (1, 0), (0, 2), (2, 0)]))
CX_H, NO_H = Target.from_configuration(["cx", "h"], 4), ["cx", "swap"]
# (gates, rejecting options or None, accepting options, operation counts); all use ``check=[]``
SUPPORT = (
    # g = CZ puts h gates in the blocks
    ((SWAP, cx(0, 2), cx(1, 3), SWAP), {"basis_gates": NO_H}, {"target": CX_H}, {"cx": 6, "h": 4}),
    # g is CX(1, 0), emitted as cx(1, 0) and cx(0, 1)
    ((swap(1, 0), cx(2, 0), cx(1, 0)), {"target": ONE_WAY}, {"target": TWO_WAY}, {"cx": 3}),
    ((SWAP, cx(0, 2), cx(0, 1)), None, {"basis_gates": ["cx", "h"]}, {"cx": 3}),
)

ANGLES = (1e-9, np.pi - 1e-9, np.pi, -np.pi, 3 * np.pi + 0.1, -7.0)


class SwapAbsorptionTestCase(QiskitTestCase):
    """Shared helpers."""

    def run_pass(self, qc, move_diagonal=True, exact=True, peephole_in_loop=False):
        """Run the pass without a check; return ``(output, stats)``, asserting exactness."""
        stats = Counter()
        _rewrites(circuit_to_dag(qc), move_diagonal, False, stats, peephole_in_loop)
        pass_ = _AbsorbIntoSwaps(move_diagonal=move_diagonal, peephole_in_loop=peephole_in_loop)
        out = dag_to_circuit(pass_.run(circuit_to_dag(qc)))
        if exact:
            self.assert_exact(qc, out)
        return out, stats

    def assert_exact(self, expected, actual, atol=1e-12):
        """Same operator, global phase included, to ``atol`` with no relative tolerance."""
        diff = np.abs(Operator(expected).data - Operator.from_circuit(actual).data).max()
        self.assertLess(diff, atol, f"operators differ by {diff}")

    def assert_rewritten(self, qc, rewrites=1, **kwargs):
        """The pass applies ``rewrites`` rewrites, each saving two two-qubit gates."""
        out, stats = self.run_pass(qc, **kwargs)
        self.assertEqual(stats["applied"], rewrites, stats)
        self.assertEqual(two_qubit_weight(out), two_qubit_weight(qc) - 2 * rewrites)
        self.assertNotIn("swap", out.count_ops())
        return out

    def assert_unchanged(self, qc, reason, **kwargs):
        """The pass rewrites nothing and ``stats`` names ``reason``."""
        out, stats = self.run_pass(qc, **kwargs)
        self.assertEqual(stats["applied"], 0, stats)
        self.assertGreaterEqual(stats[reason], 1, stats)
        self.assertEqual(out.count_ops(), qc.count_ops())


@ddt
class TestAbsorbIntoSwaps(SwapAbsorptionTestCase):
    """The rewrites, the checks, the support check and odd inputs, on the pass alone."""

    @data(*REWRITES)
    @unpack
    def test_rewrite_table(self, gates, operations):
        """Each rewrite in all four variants leaves the listed operations."""
        for qubits, step in VARIANTS:
            with self.subTest(qubits=qubits, step=step):
                out = self.assert_rewritten(circuit([*gates[::step], BARRIER], qubits=qubits))
                self.assertEqual(out.count_ops(), {**operations, "barrier": 1})

    def test_gap_with_single_qubit_gates(self):
        """Single-qubit gates in between pick g = CX(0, 1) for a later SWAP."""
        # Z-type gates on wire 0 and an X-type gate on wire 1; C3 is checked at both far ends.
        gates = [SWAP, RZ, cx(0, 2), ("t", (0,)), cx(3, 1), ("sx", (1,)), swap(1, 0)]
        out = self.assert_rewritten(circuit([*gates, cx(0, 3)]))
        self.assertEqual(out.count_ops()["cx"], 7)

    def test_diagonal_angles_and_initial_phase(self):
        """Diagonal partners at boundary angles, with a nonzero initial global phase."""
        for gate, angle in itertools.product((CPhaseGate, CU1Gate, CRZGate, RZZGate), ANGLES):
            for qubits, step in VARIANTS:
                with self.subTest(gate=gate.__name__, angle=angle, qubits=qubits, step=step):
                    qc = circuit([SWAP, cx(0, 2), cx(1, 3)][::step], qubits=qubits)
                    qc.compose(gate(angle), [qubits[1], qubits[0]], front=step < 0, inplace=True)
                    qc.barrier()
                    qc.global_phase = 0.3
                    self.assert_rewritten(qc)

    def test_two_rewrites_with_distinct_angles_sharing_a_neighbourhood(self):
        """Two SWAPs on one pair, each merged with its own diagonal gate."""
        # Each partner lies in the other rewrite's C3 walk (decisions on the original DAG).
        gates = [SWAP, cx(0, 2), ("cp", (1, 0), 0.7), cx(0, 3), ("cp", (1, 0), 1.9), cx(0, 2)]
        self.assert_rewritten(circuit([*gates, swap(1, 0), cx(0, 2), cx(1, 3)]), rewrites=2)

    @data(*CHECKS)
    @unpack
    def test_check(self, reason, before, rejected, control, after, move_diagonal=True):
        """The check ``reason`` leaves the rejected circuit alone; the control is rewritten."""
        qc = circuit(before + rejected + after)
        self.assert_unchanged(qc, reason, move_diagonal=move_diagonal)
        self.assert_rewritten(circuit(before + control + after), move_diagonal=move_diagonal)

    @data(*ONE_REWRITE)
    @unpack
    def test_one_rewrite_each(self, gates, stats_with, applied_without):
        """A gate is in one rewrite at most; ``peephole_in_loop`` skips the loop's pairs."""
        self.assertEqual(dict(self.run_pass(circuit(gates), peephole_in_loop=True)[1]), stats_with)
        self.assertEqual(self.run_pass(circuit(gates))[1]["applied"], applied_without)

    def test_c3_cz_native_swap_end(self):
        """On a cz-native target only the second SWAP of two is rewritten with g = CZ."""
        # A SWAP's translation ends with a bare cz there, so a following same-pair cz cancels with
        # it in the stock flow; the rewrite of the two SWAPs with g = CX would end with cx instead.
        qc = circuit([SWAP, cx(0, 2), cx(3, 1), SWAP, cx(0, 2), cz(0, 1), cx(1, 3)])
        self.assert_rewritten(qc)
        cz_native = _AbsorbIntoSwaps(basis_gates=CZ_BASIS)
        self.assertTrue(cz_native.cz_native)
        stats = Counter()
        _rewrites(circuit_to_dag(qc), True, True, stats)
        self.assertEqual(dict(stats), {"C3": 2, "applied": 1})  # only the second SWAP takes the cz
        out = dag_to_circuit(cz_native.run(circuit_to_dag(qc)))
        self.assertEqual(out.count_ops()["swap"], 1)
        self.assertNotIn("cz", out.count_ops())
        self.assert_exact(qc, out)

    @data(*SUPPORT)
    @unpack
    def test_support_check(self, gates, rejecting, accepting, expected):
        """Blocks the target or basis cannot express are vetoed; supported ones are rewritten."""
        qc = circuit(gates)
        if rejecting:
            same = dag_to_circuit(_AbsorbIntoSwaps(check=[], **rejecting).run(circuit_to_dag(qc)))
            self.assertEqual(same.count_ops(), qc.count_ops())
        out = dag_to_circuit(_AbsorbIntoSwaps(check=[], **accepting).run(circuit_to_dag(qc)))
        self.assertEqual(out.count_ops(), expected)
        self.assert_exact(qc, out)

    def test_check_errors(self):
        """The check needs a target or basis gates; a check that raises rejects the blocks."""
        with self.assertRaises(TranspilerError):
            _AbsorbIntoSwaps(check=[])
        check = [BasisTranslator(SessionEquivalenceLibrary, ["rz"])]
        absorb = _AbsorbIntoSwaps(basis_gates=["rz"], check=check)
        block = ([(CXGate(), (0, 1))], 0.0)  # (operations, global phase) on qubits (0, 1)
        self.assertIs(absorb._supported(2, [(block, (0, 1))]), False)

    def test_unbound_parameter_in_gap_compared_after_binding(self):
        """An unbound parameter of a gate in between is compared after binding."""
        qc = circuit([SWAP, ("rz", (0,), Parameter("theta")), cx(0, 2), cx(0, 1)])
        out = self.assert_rewritten(qc, exact=False)
        self.assert_exact(qc.assign_parameters([THETA]), out.assign_parameters([THETA]))

    def test_unbound_diagonal_partner_is_left_alone(self):
        """A diagonal partner with an unbound angle is not moved."""
        qc = circuit([SWAP, cx(0, 2), cx(1, 3), ("cp", (1, 0), Parameter("theta"))])
        self.assert_unchanged(qc, "partner", exact=False)

    def test_control_flow_blocks_rewrites(self):
        """Control flow in place of the later gate or in between blocks the rewrite."""
        qc = QuantumCircuit(3, 1)
        qc.swap(0, 1)
        qc.cx(0, 2)
        with qc.if_test((qc.clbits[0], 0)):
            qc.cx(0, 1)
        self.assert_unchanged(qc, "partner", exact=False)
        gapped = QuantumCircuit(3, 1)
        gapped.swap(0, 1)
        with gapped.if_test((gapped.clbits[0], 0)):
            gapped.x(0)
        gapped.cx(0, 1)
        self.assert_unchanged(gapped, "C1", exact=False)

    def test_classical_bits_and_vars(self):
        """Measurements and classical variables survive a rewrite; ours is the second store."""
        qc = QuantumCircuit(3, 1)
        flag = qc.add_var("flag", expr.lift(False))
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 1)
        qc.measure(2, 0)
        qc.store(flag, expr.lift(True))
        out = self.assert_rewritten(qc, exact=False)
        self.assertEqual(out.count_ops(), {"cx": 3, "measure": 1, "store": 2})

    def test_no_swap_returns_dag(self):
        """A circuit without SWAP gates comes back as the same DAG."""
        dag = circuit_to_dag(circuit([cx(0, 1)]))
        self.assertIs(_AbsorbIntoSwaps().run(dag), dag)


@ddt
class TestPresets(SwapAbsorptionTestCase):
    """Test the pass in the preset pass managers."""

    def absorb_passes(self, config, level, method="default"):
        """The absorption passes that the translation stage runs on a 2-qubit SWAP."""
        passes, plugins = [], PassManagerStagePluginManager()
        stage = plugins.get_passmanager_stage("translation", method, config, level)
        stage.run(circuit([SWAP], 2), callback=lambda **kwargs: passes.append(kwargs["pass_"]))
        return [pass_ for pass_ in passes if isinstance(pass_, _AbsorbIntoSwaps)]

    @idata(itertools.product(LEVELS, METHODS, BASES))
    @unpack
    def test_translation_stage(self, level, method, basis):
        """The translation stages add the pass at levels 2 and 3 unless swap is native."""
        found = self.absorb_passes(PassManagerConfig(target=line_target(2, basis)), level, method)
        present = level in (2, 3) and "swap" not in basis
        expected = [("rzz" not in basis, level == 3)] if present else []
        self.assertEqual(
            [(pass_.move_diagonal, pass_.peephole_in_loop) for pass_ in found], expected
        )

    @data(
        ({"basis_gates": CZ_BASIS}, True),
        ({}, False),
        ({"target": line_target(2), "unitary_synthesis_method": "sk"}, False),
        ({"target": line_target(2), "unitary_synthesis_plugin_config": {}}, False),
        ({"target": line_target(2), "hls_config": HLSConfig()}, False),
    )
    @unpack
    def test_translation_stage_options(self, options, present):
        """Basis gates alone suffice; no target or basis, or synthesis options, skip the pass."""
        self.assertEqual(bool(self.absorb_passes(PassManagerConfig(**options), 2)), present)

    @data(2, 3)
    def test_one_way_ecr_target(self, level):
        """A move on a one-way ECR target needs fewer two-qubit gates and keeps the direction."""
        coupling_map = CouplingMap([(0, 1), (0, 2), (1, 2)])
        target = Target.from_configuration(["ecr", "rz", "sx", "x"], 3, coupling_map)
        stage = generate_preset_pass_manager(level, target=target).translation
        # The gate in between on qubit 0 keeps the swap and the cx apart; otherwise C1 applies.
        qc = circuit([SWAP, cx(0, 2), cx(0, 1), cx(1, 2), cx(0, 1)], 3)
        qc.ensure_physical()
        out = stage.run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            self.assertLess(out.num_nonlocal_gates(), stage.run(qc).num_nonlocal_gates())
        check_direction = CheckGateDirection(None, target=target)
        check_direction(out)
        self.assertTrue(check_direction.property_set["is_direction_mapped"])
        self.assert_exact(qc, out, atol=1e-10)  # synthesis round-off downstream of the pass

    def test_qft(self):
        """A QFT absorbs routing swaps and keeps its operator, global phase included."""
        qc = QuantumCircuit(6)
        qc.append(QFTGate(6), range(6))
        options = {"basis_gates": CZ_BASIS, "coupling_map": CouplingMap.from_line(6)}
        pm = generate_preset_pass_manager(2, seed_transpiler=1, **options)
        counts = []
        out = pm.run(qc, callback=lambda pass_, dag, **_: counts.append((pass_, dag.count_ops())))
        index = [type(pass_) for pass_, _ in counts].index(_AbsorbIntoSwaps)
        self.assertLess(counts[index][1].get("swap", 0), counts[index - 1][1].get("swap", 0))
        self.assert_exact(qc, out, atol=1e-10)  # synthesis round-off downstream of the pass


if __name__ == "__main__":
    unittest.main()
