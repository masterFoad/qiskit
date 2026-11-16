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

"""Tests of the _ResynthesizeBlockPairs pass."""

import unittest
from unittest.mock import patch

import numpy as np
from ddt import data, ddt, unpack

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import CXGate, HGate, RZGate, SXGate, TGate, UGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager, PassManagerConfig, Target
from qiskit.transpiler.passes import ConsolidateBlocks, UnitarySynthesis
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager

from qiskit.transpiler.preset_passmanagers._block_pairs import _ResynthesizeBlockPairs

from test import QiskitTestCase

# Largest entry of the difference of two operators that still counts as equal; no relative
# tolerance, the global phase included.
ATOL = 1e-10


def pack(circuit, **kwargs):
    """The circuit after the pass."""
    return PassManager([_ResynthesizeBlockPairs(**kwargs)]).run(circuit)


def cx_count(circuit):
    """The CX gates left after every two-qubit block is consolidated and synthesised."""
    basis = ["cx", "u"]
    passes = [ConsolidateBlocks(basis_gates=basis, force_consolidate=True), UnitarySynthesis(basis)]
    return PassManager(passes).run(circuit).count_ops().get("cx", 0)


def two_qubit_count(circuit):
    """Count two-qubit gates, including those inside control-flow blocks."""
    return sum(
        int(isinstance(instruction.operation, Gate) and instruction.operation.num_qubits == 2)
        + sum(two_qubit_count(block) for block in getattr(instruction.operation, "blocks", ()))
        for instruction in circuit.data
    )


def build(gates, num_qubits=3, num_clbits=0):
    """A circuit from ``(name, qubits, *params)`` tuples."""
    circuit = QuantumCircuit(num_qubits, num_clbits)
    for name, qubits, *params in gates:
        getattr(circuit, name)(*params, *qubits)
    return circuit


SWAP = ("swap", (0, 1))
CX01 = ("cx", (0, 1))
CX02 = ("cx", (0, 2))
CX20 = ("cx", (2, 0))
CZ01 = ("cz", (0, 1))
RX1 = ("rx", (1,), 0.7)

# (qubits, gates, CX count before, CX count after, operations after)
PACKED_COUNTS = (
    # a SWAP and a later CX on the same qubits: ``swap cx`` is two CX, ``cx cx`` none
    (3, (SWAP, CX02, CX01), 5, 3, {"unitary": 2, "cx": 1}),
    # two SWAPs across a gap share a CX frame: two CX each
    (3, (SWAP, CX02, SWAP), 7, 5, {"unitary": 2, "cx": 1}),
    # the partner may come first
    (3, (CX01, CX02, SWAP), 5, 3, {"unitary": 2, "cx": 1}),
    # ``cz rx cz`` blocks on both sides of a gap that commutes with Z lose one CZ each
    (3, (CZ01, RX1, CZ01, ("cz", (0, 2)), CZ01, RX1, CZ01), 5, 3, {"unitary": 2, "cz": 1}),
    # a gap that commutes with X on the shared wire takes the frame ``cx(1, 0)``
    (3, (SWAP, CX20, ("cx", (1, 0))), 5, 3, {"unitary": 2, "cx": 1}),
    # a gap that commutes with Y on the shared wire takes the frame ``cy(1, 0)``
    (3, (SWAP, ("cy", (2, 0)), ("cy", (1, 0))), 5, 3, {"unitary": 2, "cy": 1}),
    # both wires may carry gap gates, each with its own Pauli
    (4, (SWAP, CX02, ("cx", (3, 1)), CX01), 6, 4, {"unitary": 2, "cx": 2}),
    # single-qubit gates next to the blocks do not stop the rewrite
    (3, (SWAP, ("t", (0,)), CX02, ("sx", (0,)), CX01), 5, 4, {"unitary": 2, "cx": 1}),
    # a block that was rewritten can be rewritten again with the next block
    (3, (SWAP, CX02, SWAP, CX02, SWAP, CX02, SWAP, CX02, CX01), 17, 10, {"unitary": 5, "cx": 4}),
    # a candidate whose CX count ``num_basis_gates`` gets wrong is checked by synthesis
    (3, (SWAP, ("h", (0,)), ("cz", (0, 2)), ("h", (0,)), SWAP), 7, 5, {"unitary": 2, "cz": 1}),
    # no rewrite when the gap gates share no Pauli on a wire, or when no frame lowers the cost
    (3, (SWAP, CX02, CX20, CX01), 6, 6, {"swap": 1, "cx": 3}),
    (3, (CX01, CX20, CX01), 3, 3, {"cx": 3}),
)

# Operations that end a walk: not bound unitary gates on at most two qubits
WALK_ENDS = (
    ("delay", (0,), 100),
    ("barrier", (0,)),
    ("barrier", (0, 1, 2, 3)),
    ("measure", (0, 0)),
    ("reset", (0,)),
    ("rz", (0,), Parameter("a")),
    ("rzz", (0, 2), Parameter("b")),
    ("ccx", (0, 2, 3)),
)

# (upper bound of the random draw, gate, takes an angle, acts on the pair)
RANDOM_GATES = (
    (0.2, "swap", False, True),
    (0.5, "cx", False, True),
    (0.58, "cz", False, True),
    (0.66, "rzz", True, True),
    (0.72, "cp", True, True),
    (0.8, "h", False, False),
    (0.9, "rz", True, False),
    (1.0, "sx", False, False),
)


@ddt
class TestResynthesizeBlockPairs(QiskitTestCase):
    """Tests of the _ResynthesizeBlockPairs pass."""

    def assertSameOperator(self, first, second):
        """Assert that two circuits have the same operator, phase included."""
        difference = np.abs(Operator(first).data - Operator(second).data).max()
        self.assertLessEqual(difference, ATOL)

    def assertNoUnitary(self, circuit):
        """No unsynthesized unitary remains, including in control-flow blocks."""
        self.assertNotIn("unitary", circuit.count_ops())
        for instruction in circuit.data:
            for block in getattr(instruction.operation, "blocks", ()):
                self.assertNoUnitary(block)

    def assertSameLogicalOperator(self, first, second):
        """Compare phase-inclusive operators after applying any transpilation layouts."""
        difference = np.abs(
            Operator.from_circuit(first).data - Operator.from_circuit(second).data
        ).max()
        self.assertLessEqual(difference, ATOL)

    def assertPresetResult(self, circuit, actual, stock):
        """The preset returns an exact native circuit with no extra two-qubit gates."""
        self.assertNoUnitary(actual)
        self.assertLessEqual(two_qubit_count(actual), two_qubit_count(stock))
        self.assertSameLogicalOperator(circuit, actual)

    @data(*PACKED_COUNTS)
    @unpack
    def test_packed_counts(self, num_qubits, gates, before, after, operations):
        """The CX count goes from ``before`` to ``after``, the operator is kept."""
        circuit = build(gates, num_qubits)
        packed = pack(circuit)
        self.assertSameOperator(circuit, packed)
        self.assertEqual((cx_count(circuit), cx_count(packed)), (before, after))
        self.assertEqual(packed.count_ops(), operations)

    def test_gate_with_definition_only(self):
        """A gate without ``to_matrix`` is read through its definition."""
        definition = QuantumCircuit(2, name="defined")
        definition.cx(0, 1)
        circuit = build([SWAP])
        circuit.append(definition.to_gate(), [0, 2])
        circuit.cx(0, 1)
        packed = pack(circuit)
        self.assertSameOperator(circuit, packed)
        self.assertEqual((cx_count(circuit), cx_count(packed)), (5, 3))

    def test_operations_that_end_the_walk(self):
        """Nothing is rewritten across an operation that is not a bound unitary gate."""
        opaque = QuantumCircuit(4, 1)
        opaque.append(Gate("opaque", 1, []), [0])
        if_test = QuantumCircuit(4, 1)
        with if_test.if_test((if_test.clbits[0], 1)):
            if_test.z(0)
        for blocker in [build([row], 4, 1) for row in WALK_ENDS] + [opaque, if_test]:
            for before, after in (([SWAP], [CX02, CX01]), ([SWAP, CX02], [CX01])):
                with self.subTest(blocker=blocker.data[0].name, before=before):
                    circuit = build(before, 4, 1).compose(blocker).compose(build(after, 4, 1))
                    self.assertEqual(pack(circuit), circuit)

    def test_operations_off_the_pair_are_kept(self):
        """Operations on other wires neither block the rewrite nor change."""
        off_pair = [("delay", (3,), 50), ("reset", (3,)), ("rz", (3,), Parameter("a"))]
        circuit = build([SWAP, *off_pair, CX02, ("barrier", (3,)), CX01, ("measure", (3, 1))], 4, 2)
        off_pair_ops = {"delay": 1, "reset": 1, "rz": 1, "barrier": 1, "measure": 1}
        self.assertEqual(pack(circuit).count_ops(), {"unitary": 2, "cx": 1, **off_pair_ops})

    def test_final_measurements(self):
        """Measurements after the partner do not stop the rewrite."""
        unitary = build([SWAP, CX02, CX01], 3, 3)
        circuit = unitary.copy()
        circuit.measure([0, 1, 2], [0, 1, 2])
        packed = pack(circuit)
        self.assertEqual(packed.count_ops(), {"unitary": 2, "cx": 1, "measure": 3})
        self.assertSameOperator(unitary, packed.remove_final_measurements(inplace=False))

    def test_target(self):
        """The pass runs for cx, cz and ecr targets and leaves other targets alone."""
        circuit = build([SWAP, CX02, CX01])
        cases = ((["cx"], True), (["cz"], True), (["ecr"], True), (["cz", "rzz"], False))
        for two_qubit, runs in cases:
            with self.subTest(two_qubit=two_qubit):
                basis = ["id", "rz", "sx", "x", *two_qubit]
                target = GenericBackendV2(3, basis_gates=basis, seed=1, control_flow=True).target
                packed = pack(circuit, target=target)
                self.assertEqual(packed != circuit, runs)
                self.assertSameOperator(circuit, packed)

    def test_random_circuits(self):
        """Random nearest-neighbour circuits: same operator, never more CX gates."""
        rng = np.random.default_rng(1234)
        improved = 0
        for index in range(150):
            num_qubits = int(rng.integers(3, 6))
            circuit = QuantumCircuit(num_qubits)
            for _ in range(int(rng.integers(8, 50))):
                first = int(rng.integers(num_qubits - 1))
                pair = (first, first + 1) if rng.random() < 0.5 else (first + 1, first)
                choice = rng.random()
                _, name, angle, on_pair = next(row for row in RANDOM_GATES if choice < row[0])
                params = [rng.normal()] if angle else []
                getattr(circuit, name)(*params, *(pair if on_pair else (first,)))
            with self.subTest(index=index):
                packed = pack(circuit)
                self.assertSameOperator(circuit, packed)
                before, after = cx_count(circuit), cx_count(packed)
                self.assertLessEqual(after, before)
                improved += after < before
        self.assertGreater(improved, 50)

    def test_second_run_changes_nothing_for_the_worse(self):
        """Running the pass on its own output keeps the operator and the CX count."""
        circuit = QuantumCircuit(4)
        for first in (0, 1, 2, 0, 1, 2):
            circuit.swap(first, first + 1)
            circuit.cx(first, (first + 2) % 4)
            circuit.cx(first, first + 1)
        once = pack(circuit)
        twice = pack(once)
        self.assertSameOperator(circuit, twice)
        self.assertLessEqual(cx_count(twice), cx_count(once))

    def test_preset_wiring(self):
        """Paired resynthesis runs after optimization at O2/O3 for CX-equivalent targets only."""
        circuit = build([SWAP, CX02, CX01])
        circuit.global_phase = 0.3
        for level, basis, enabled in (
            (1, ["cx"], False),
            (2, ["cx"], True),
            (3, ["cx"], True),
            (2, ["cz"], True),
            (3, ["ecr"], True),
            (2, ["cz", "rzz"], False),
            (3, ["cz", "rzz"], False),
        ):
            with self.subTest(level=level, basis=basis):
                target = GenericBackendV2(3, basis_gates=["u", *basis], seed=1).target
                manager = generate_preset_pass_manager(
                    level, target=target, seed_transpiler=11, initial_layout=[0, 1, 2]
                )
                trace = []
                actual = manager.run(
                    circuit, callback=lambda **kwargs: trace.append(kwargs["pass_"].name())
                )
                self.assertEqual(trace.count("_ResynthesizeBlockPairs"), int(enabled))
                self.assertEqual(manager.post_optimization is not None, enabled)
                if enabled:
                    position = trace.index("_ResynthesizeBlockPairs")
                    self.assertIn("CommutativeCancellation", trace[:position])
                    self.assertEqual(
                        trace[position + 1 : position + 3], ["_BlockPairReplay", "_BlockPairReplay"]
                    )
                manager.post_optimization = None
                stock = manager.run(circuit)
                self.assertPresetResult(circuit, actual, stock)

    def test_preset_matches_separate_post_runs(self):
        """The wired post stage matches fresh translation and optimization runs, including phase."""
        circuit = build([SWAP, CX02, SWAP, CX02, CX01])
        circuit.global_phase = 0.3
        target = GenericBackendV2(3, basis_gates=["u", "cx"], seed=1).target
        for level in (2, 3):
            with self.subTest(level=level):
                manager = generate_preset_pass_manager(
                    level, target=target, seed_transpiler=11, initial_layout=[0, 1, 2]
                )
                actual = manager.run(circuit)
                manager.post_optimization = None
                stock = manager.run(circuit)
                translated = manager.translation.run(pack(stock, target=target))
                # The explicit initial layout disables post-layout search. Run the original
                # optimization stage directly as an independent oracle, without copying a filter.
                expected = manager.optimization.run(translated)
                self.assertEqual(actual.data, expected.data)
                self.assertEqual(actual.global_phase, expected.global_phase)
                self.assertPresetResult(circuit, actual, stock)

    def test_non_default_unitary_synthesis_disables_stage(self):
        """Solovay-Kitaev does not satisfy the pass's exact synthesis assumption."""
        manager = generate_preset_pass_manager(
            3, basis_gates=["u", "cx"], unitary_synthesis_method="sk"
        )
        self.assertIsNone(manager.post_optimization)

    def test_external_translation_disables_stage(self):
        """An external translation stage is not replayed under the default synthesis contract."""
        original = PassManagerStagePluginManager.get_passmanager_stage

        def external_translation(plugin_manager, stage, name, config, optimization_level=None):
            return original(
                plugin_manager,
                stage,
                "translator" if stage == "translation" and name == "external" else name,
                config,
                optimization_level,
            )

        with patch.object(
            PassManagerStagePluginManager, "get_passmanager_stage", external_translation
        ):
            manager = generate_preset_pass_manager(
                3, basis_gates=["u", "cx"], translation_method="external"
            )
        self.assertIsNone(manager.post_optimization)

    def test_discrete_basis_disables_stage(self):
        """Clifford+T does not exactly synthesize arbitrary single-qubit unitaries."""
        manager = generate_preset_pass_manager(3, basis_gates=["cx", "h", "t"])
        self.assertIsNone(manager.post_optimization)

    @data(0.0, 0.9)
    def test_approximation_disables_stage(self, degree):
        """Explicit approximate synthesis is outside the pass's certification policy."""
        manager = generate_preset_pass_manager(
            3, basis_gates=["u", "cx"], approximation_degree=degree
        )
        self.assertIsNone(manager.post_optimization)

    def test_continuous_basis_on_each_qubit(self):
        """Aggregate target names must not hide a qubit with only a discrete basis."""
        target = Target(num_qubits=3)
        target.add_instruction(
            UGate(*(Parameter(name) for name in ("theta", "phi", "lambda"))),
            {(0,): None, (2,): None},
        )
        target.add_instruction(HGate(), {(1,): None})
        target.add_instruction(TGate(), {(1,): None})
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None})
        manager = generate_preset_pass_manager(3, target=target)
        self.assertIsNone(manager.post_optimization)

    def test_missing_single_qubit_basis_disables_stage(self):
        """A physical qubit without single-qubit instructions disables the stage safely."""
        target = Target(num_qubits=3)
        target.add_instruction(
            UGate(*(Parameter(name) for name in ("theta", "phi", "lambda"))),
            {(0,): None, (2,): None},
        )
        target.add_instruction(CXGate(), {(0, 2): None})
        for level in (2, 3):
            with self.subTest(level=level):
                manager = generate_preset_pass_manager(level, target=target)
                self.assertIsNone(manager.post_optimization)

    def test_fixed_and_restricted_rotations_disable_stage(self):
        """A rotation name with fixed or restricted parameters is not a free Euler rotation."""
        for rotation, bounds in (
            (RZGate(0.5), None),
            (RZGate(Parameter("theta")), [(-0.5, 0.5)]),
        ):
            with self.subTest(rotation=rotation, bounds=bounds):
                target = Target(num_qubits=3)
                target.add_instruction(rotation, angle_bounds=bounds)
                target.add_instruction(SXGate())
                target.add_instruction(CXGate())
                manager = generate_preset_pass_manager(3, target=target)
                self.assertIsNone(manager.post_optimization)

    def test_basis_gates_only_wiring(self):
        """The public basis-gates-only preset call enables the stage and returns exact gates."""
        circuit = build([SWAP, CX02, SWAP, CX02, CX01])
        circuit.global_phase = 0.3
        for level in (2, 3):
            with self.subTest(level=level):
                manager = generate_preset_pass_manager(level, basis_gates=["u", "cx"])
                self.assertIsNotNone(manager.post_optimization)
                actual = manager.run(circuit)
                manager.post_optimization = None
                self.assertPresetResult(circuit, actual, manager.run(circuit))

    def test_direct_level_manager_without_target(self):
        """Direct level-manager calls support a standard continuous basis with target=None."""
        for factory in (level_2_pass_manager, level_3_pass_manager):
            for degree in (None, 1.0):
                with self.subTest(factory=factory.__name__, degree=degree):
                    config = PassManagerConfig(
                        basis_gates=["u", "cx"], approximation_degree=degree, seed_transpiler=11
                    )
                    self.assertIsNone(config.target)
                    manager = factory(config)
                    self.assertIsNotNone(manager.post_optimization)

    def test_non_default_optimization_disables_stage(self):
        """A custom optimization stage is not replayed."""
        original = PassManagerStagePluginManager.get_passmanager_stage

        def external_optimization(plugin_manager, stage, name, config, optimization_level=None):
            if stage == "optimization" and name == "external":
                return PassManager()
            return original(plugin_manager, stage, name, config, optimization_level)

        with patch.object(
            PassManagerStagePluginManager, "get_passmanager_stage", external_optimization
        ):
            manager = generate_preset_pass_manager(
                3, basis_gates=["u", "cx"], optimization_method="external"
            )
        self.assertIsNone(manager.post_optimization)
        circuit = build([SWAP, CX02, CX01])
        actual = manager.run(circuit)
        self.assertNoUnitary(actual)
        self.assertSameLogicalOperator(circuit, actual)

    def test_preset_control_flow(self):
        """Control flow survives the replay, with exact operators for both classical inputs."""
        circuit = build([SWAP, CX02, CX01], num_clbits=1)
        circuit.global_phase = 0.3
        with circuit.if_test((circuit.clbits[0], 1)):
            circuit.h(2)
            circuit.cx(0, 2)

        def branch(circuit, condition):
            result = circuit.copy_empty_like()
            for instruction in circuit.data:
                if instruction.operation.name == "if_else":
                    self.assertEqual(instruction.operation.condition, (circuit.clbits[0], 1))
                    block_index = 0 if condition else 1
                    if block_index < len(instruction.operation.blocks):
                        result.compose(
                            instruction.operation.blocks[block_index],
                            instruction.qubits,
                            instruction.clbits,
                            inplace=True,
                        )
                else:
                    result.append(instruction)
            return result

        target = GenericBackendV2(3, basis_gates=["u", "cx"], control_flow=True, seed=1).target
        for level in (2, 3):
            with self.subTest(level=level):
                manager = generate_preset_pass_manager(
                    level, target=target, initial_layout=[0, 1, 2], seed_transpiler=11
                )
                self.assertIsNotNone(manager.post_optimization)
                actual = manager.run(circuit)
                manager.post_optimization = None
                stock = manager.run(circuit)
                self.assertNoUnitary(actual)
                self.assertLessEqual(two_qubit_count(actual), two_qubit_count(stock))
                self.assertEqual(actual.count_ops().get("if_else"), 1)
                for condition in (False, True):
                    self.assertSameLogicalOperator(
                        branch(circuit, condition), branch(actual, condition)
                    )


if __name__ == "__main__":
    unittest.main()
