"""Small-cell suppression over additive constraints (the export's one masking rule).

A table that is exported is a set of counts (cells), and many of them add up:
a whole-slice count is the sum of its subgroup levels, overall is dev plus
test, a stage's exclusions are the sum of its reasons. Masking a small cell is
useless if a total and the other parts are shown, because the masked value is
their difference. This module decides which cells to hide for *all* the
tables of one bundle at once, from a list of such constraints.

Rule:

1. **Primary.** Every cell with ``0 < count < min_cell`` is hidden.
2. **Complementary, to a fixed point.** For every constraint
   ``total = part_1 + ... + part_k``:

   - total shown: the hidden parts are either none, or at least two whose
     sum is at least ``min_cell`` (their sum is ``total - shown parts``);
   - total hidden: not every part is shown (the total would be their sum).

   A violated constraint hides its smallest shown positive part, or the
   total when no part is left to hide, and the check runs again.
3. **Audit.** Constraints combine: a reader may solve several at once. For
   every hidden cell whose value is 1 to ``min_cell - 1`` the feasible range
   ``[lo, hi]`` over the whole constraint system is computed by linear
   programming, under the attacker model below. The cell is protected when
   ``hi >= min_cell`` and ``hi - lo >= min_cell - 1``. An unprotected cell
   makes one more cell hidden (the smallest shown cell sharing a constraint
   with it), and rules 2 and 3 run again.
4. **Prune.** At a clean fixed point, each complement the caller marks
   (:meth:`Cells.prefer_together`: the export's whole slices, a row's
   cells together) is shown again when the system
   without it still passes rule 3 and gives no small count away. The state
   returned is audited once more from scratch.

Attacker model: every shown cell and every fixed cell is known, every
constraint is known, and every hidden cell is an integer of at least 1
(zero is never hidden, so it is shown as 0), a latent cell (below) one of at
least 0; the audit uses the continuous relaxation.

What a reader may believe beyond the files must be in the system to be
covered. Scored counts are treated as reader-known relative to Table 1:
wherever an arm scored every ECG of a Table 1 count (per split and level),
the export states scored = that count, so a reader who takes every arm to
have scored every Table 1 ECG (true at a uniform site) gains nothing the
audit has not used. The scored count stays a latent cell, so a small Table 1
level the reader believes fully scored is as hidden as the level; a gap of
1 or more (ECGs an arm did not score) stays unknown, as a count no file
shows, the way a zero is public and a small count is not. Beliefs about *which* records
(two arms scored identical record sets, so their joint-label counts or a
comparison's pair counts follow) are set-level, not counts, and are out of
scope for v1.

A cell may be *fixed*: known outside the bundle's control (a stage total
that the record counts before and after it give away), so it is never hidden
and only its parts can be. A table where only a fixed cell could protect a
small cell is refused (:class:`DisclosureError`) unless the caller asks for
the unprotected cells instead.

A constraint need not be exact: :meth:`Cells.constrain` adds the remainder
``total - sum(parts)`` as one more (never displayed) cell, so records outside
every part (unlabelled ECGs, say) are protected like a part. :meth:`Cells.equal`
links two cells counting the same records in two tables: they are hidden
together, and they enter the audit as equal only when their values are.
:meth:`Cells.at_most` states that some cells count a subset of another's
records (an arm's evaluated ECGs within Table 1's): the audit adds
``sum(parts) <= total``, and an equality when the counts agree, since a
reader may then take them for the same records.

Cells whose small values the caller accepts as unprotectable (a stage total
the site allowed to leave) are :attr:`Cells.tolerated`: a strict solve does
not refuse over them.

A cell may be *latent* (:meth:`Cells.latent`): a count no table shows and the
bundle does not know (the labelled ECGs of one split and subgroup level, say),
which a reader can still bound through the relations it is in. Latent cells
are always hidden, are at least 0 in the audit, and take part only through
relations stated with :meth:`Cells.define` and :meth:`Cells.at_most`, never
the complementary rule. As their value is unknown, the audit treats every
latent cell a reader could pin to 1 to ``min_cell - 1`` as a small cell: its
feasible range must either miss that interval or protect it.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

PRIMARY = "primary"
COMPLEMENT = "complement"
#: Mark of a latent cell in :meth:`Cells.solve`'s result (never displayed).
LATENT = "latent"
#: Largest upper bound of a hidden cell in the audit's linear programs (see _program).
_CAP = 1e9


class DisclosureError(Exception):
    """A file about to be exported holds, or would let a reader recover, a record-level value."""


class Cells:
    """Counts and the additive constraints between them."""

    def __init__(self, name: str = "table") -> None:
        self.name = name
        self.values: List[float] = []
        self.fixed: List[bool] = []
        self.constraints: List[Tuple[int, Tuple[int, ...]]] = []
        #: ``equal`` pairs: hidden together; numeric only when the values agree.
        self.links: List[Tuple[int, Tuple[int, ...]]] = []
        #: ``at_most`` relations: ``sum(parts) <= total``, for the audit only.
        self.subsets: List[Tuple[int, Tuple[int, ...]]] = []
        self.forced: Set[int] = set()
        self.tolerated: Set[int] = set()
        self.latent_cells: Set[int] = set()
        self.latent_audited: Set[int] = set()
        self.unprotected: List[int] = []
        #: Groups of cells :meth:`_prune` tries to show again, each as one.
        self.prefer: List[Tuple[int, ...]] = []
        self._index: Dict[Hashable, int] = {}

    def add(self, value, key: Optional[Hashable] = None, fixed: bool = False) -> Optional[int]:
        """A cell; ``key`` returns the existing cell for the same key. NaN is no cell."""
        if key is not None and key in self._index:
            return self._index[key]
        number = _number(value)
        if number is None:
            return None
        self.values.append(number)
        self.fixed.append(bool(fixed))
        index = len(self.values) - 1
        if key is not None:
            self._index[key] = index
        return index

    def latent(self, key: Optional[Hashable] = None, audited: bool = True) -> int:
        """A count no table shows and whose value is unknown (see the module docstring).

        ``audited=False`` is a helper (a sum of other cells) whose own range
        needs no check.
        """
        if key is not None and key in self._index:
            return self._index[key]
        self.values.append(math.nan)
        self.fixed.append(False)
        index = len(self.values) - 1
        self.latent_cells.add(index)
        if audited:
            self.latent_audited.add(index)
        if key is not None:
            self._index[key] = index
        return index

    def prefer_together(self, cells: Sequence[Optional[int]]) -> None:
        """Cells worth showing again together (one table row): see :meth:`_prune`."""
        group = tuple(sorted({c for c in cells if c is not None}))
        if group:
            self.prefer.append(group)

    def get(self, key: Hashable) -> Optional[int]:
        return self._index.get(key)

    def define(self, total: Optional[int], parts: Sequence[Optional[int]]) -> None:
        """``total = sum(parts)`` by definition, where some member is latent.

        Such a relation is for the audit only: the complementary rule cannot
        weigh a sum whose value is unknown.
        """
        parts = tuple(p for p in parts if p is not None)
        if total is None or not parts:
            return
        self.constraints.append((total, parts))

    def _has_latent(self, total, parts) -> bool:
        return total in self.latent_cells or any(p in self.latent_cells for p in parts)

    def constrain(self, total: Optional[int], parts: Sequence[Optional[int]]) -> None:
        """``total = sum(parts)``, plus a remainder cell when the parts fall short."""
        parts = tuple(p for p in parts if p is not None)
        if total is None or not parts:
            return
        remainder = self.values[total] - sum(self.values[p] for p in parts)
        if remainder > 1e-9:
            parts = parts + (self.add(remainder),)
        self.constraints.append((total, parts))

    def equal(self, a: Optional[int], b: Optional[int]) -> None:
        """Two cells counting the same records: hidden in both or shown in both."""
        if a is None or b is None or a == b:
            return
        self.links.append((a, (b,)))
        self.links.append((b, (a,)))

    def at_most(self, total: Optional[int], parts: Sequence[Optional[int]]) -> None:
        """``parts`` count records that ``total`` counts too: ``sum(parts) <= total``.

        When the counts agree a reader may take them for the same records, so
        the relation is then an equality (one part: :meth:`equal`). A relation
        the counts do not satisfy says nothing true and is not added.
        """
        parts = tuple(p for p in parts if p is not None and p != total)
        if total is None or not parts:
            return
        if self._has_latent(total, parts):
            # True by definition; an unknown count is never taken for an equal one.
            self.subsets.append((total, parts))
            return
        gap = self.values[total] - sum(self.values[p] for p in parts)
        if gap < -1e-9:
            return
        if gap > 1e-9:
            self.subsets.append((total, parts))
        elif len(parts) == 1:
            self.equal(total, parts[0])
        else:
            self.constraints.append((total, parts))

    # -- solving ---------------------------------------------------------------

    def solve(self, min_cell: int, audit: bool = True, strict: bool = True,
              prune: bool = True) -> Dict[int, str]:
        """Hidden cells, each marked :data:`PRIMARY` (small) or :data:`COMPLEMENT`.

        With ``audit`` the multi-constraint audit runs after the complementary
        rule, hiding more cells until every small hidden cell is protected.
        ``strict`` raises :class:`DisclosureError` when that is impossible or a
        constraint is still violated; otherwise the cells left unprotected are
        in :attr:`unprotected`.

        ``prune`` (with ``audit``): once the fixed point is clean, complements
        that turn out not to be needed are shown again (:meth:`_prune`).

        The state returned is audited afresh, without any memo. Should it
        fail, the prune is undone and the fixed point resumes from there,
        hiding more; a strict solve refuses what is still unprotected.
        """
        prune = prune and audit
        components = self._components()
        self._check_consistent(components)
        memo: Dict = {}
        while True:
            while True:
                hidden = self._complement(min_cell)
                failing = self.audit(min_cell, hidden, components, memo) if audit else []
                if not self._hide_beside(failing, hidden):
                    break
                # Once hidden, a cell stays hidden: the hidden set only grows, so a
                # cell the audit found protected stays protected (see audit).
                self.forced.update(c for c, mark in hidden.items() if mark == COMPLEMENT)
            if not audit:
                break
            before = (dict(hidden), set(self.forced))
            if prune and not failing and not self.violations(min_cell, hidden):
                self._prune(min_cell, hidden, components)
            # Backstop: the final state, audited afresh (no memo), is what leaves.
            failing = self.audit(min_cell, hidden, components)
            if failing and hidden != before[0]:
                hidden, self.forced = before[0], before[1]   # undo the prune
                failing = self.audit(min_cell, hidden, components)
            if not failing or not self._hide_beside(failing, hidden):
                break
            self.forced.update(c for c, mark in hidden.items() if mark == COMPLEMENT)
            prune = False   # a second pass only hides; it never prunes again
        self.unprotected = failing
        if strict:
            def accepted(members):
                return all(m in self.tolerated for m in members if not self.fixed[m])

            problems = [(t, p) for t, p in self.violations(min_cell, hidden)
                        if not accepted((t,) + tuple(p))]
            failing = [c for c in failing if c not in self.tolerated]
            if failing or problems:
                cells = sorted(set(failing) | {t for t, _ in problems})
                raise DisclosureError(
                    f"{self.name}: {len(cells)} small count(s) could still be recovered and "
                    "only cells known elsewhere (fixed totals) could protect them; export refused"
                )
        return hidden

    def _prune(self, min_cell: int, hidden: Dict[int, str], components) -> None:
        """Show again each :attr:`prefer` group whose hiding protects nothing.

        The complementary rule and the audit's neighbour picks are greedy: a
        cell picked early (a large whole slice, say) can turn out redundant
        once later picks are in. A group (a table row's counts) is tried only
        when every hidden member is a complement (a row with a small count
        stays blank anyway): its hidden members, with the cells
        :meth:`equal` ties to them, are shown again only when :meth:`violations` and the audit of the touched components
        stay empty for the whole system. The audit is the gate, as in
        :meth:`solve`, so no small cell loses protection.

        Groups are tried all at once, then by halves: a batch that passes costs
        one component audit, and a failing one usually stops at its first
        target, the cells next to the batch being checked first. Only
        :attr:`prefer` is tried: every complement one by one costs a
        component audit each, a minute on a small study.
        """
        where = {c: k for k, component in enumerate(components) for c in component[0]}
        tied: Dict[int, Set[int]] = {}
        for a, (b,) in self.links:
            tied.setdefault(a, set()).add(b)
        beside: Dict[int, Set[int]] = {}
        for total, parts in self.constraints + self.links + self.subsets:
            members = (total,) + tuple(parts)
            for m in members:
                beside.setdefault(m, set()).update(members)
        memo: Dict = {}

        def unit_of(group):
            unit = {c for c in group if c in hidden}
            stack = list(unit)
            while stack:
                for other in tied.get(stack.pop(), ()):
                    if other not in unit and other in hidden:
                        unit.add(other)
                        stack.append(other)
            return unit if unit and all(hidden[c] == COMPLEMENT for c in unit) else None

        def shown_again(units) -> bool:
            cells = set().union(*units)
            trial = {c: mark for c, mark in hidden.items() if c not in cells}
            if self.violations(min_cell, trial):
                return False
            near = set().union(*(beside.get(c, set()) for c in cells))
            near = set().union(near, *(beside.get(c, set()) for c in near))
            touched = sorted({where[c] for c in cells if c in where})
            if any(self.audit(min_cell, trial, [components[k]], memo, first=True, early=near)
                   for k in touched):
                return False
            for c in cells:
                del hidden[c]
                self.forced.discard(c)
            return True

        def attempt(batch) -> None:
            # All at once, else each half: one audit for a batch that passes,
            # about log2(len) more per group that has to stay hidden.
            units = [u for u in (unit_of(g) for g in batch) if u is not None]
            if not units or shown_again(units) or len(batch) == 1:
                return
            half = len(batch) // 2
            attempt(batch[:half])
            attempt(batch[half:])

        attempt(sorted(set(self.prefer), key=lambda g: (-max(self.values[c] for c in g), g)))

    def _check_consistent(self, components) -> None:
        """Refuse a system its own true counts do not satisfy.

        Every relation is meant to be true of the data. A false one (a model
        error) leaves the audit's linear programs without any solution, so
        every range would come back without a verdict and fail closed: whole
        tables would be hidden for no reader's sake. Shown counts are checked
        directly, latent ones by one feasibility program per component that
        holds any, every non-latent cell at its true value.
        """
        def false(relation, inequality):
            total, parts = relation
            if self._has_latent(total, parts):
                return False
            gap = self.values[total] - sum(self.values[p] for p in parts)
            return gap < -1e-6 if inequality else abs(gap) > 1e-6

        bad = [r for r in self._numeric() if false(r, False)]
        bad += [r for r in self.subsets if false(r, True)]
        for component in components:
            if bad:
                break
            latent = {c for c in component[0] if c in self.latent_cells}
            if latent and not self._feasible(self._program(component, latent)):
                bad.append(next(r for r in component[1] + component[2]
                                if self._has_latent(*r)))
        if bad:
            raise DisclosureError(
                f"{self.name}: {len(bad)} relation(s) between the counts are false of the counts "
                "themselves, so the masking audit cannot run; export refused (a model error, "
                "not a property of the data)"
            )

    def _feasible(self, program) -> bool:
        import numpy as np
        from scipy.optimize import linprog

        if program["empty"]:
            return True
        objective = np.zeros(len(program["box"]))
        result = linprog(objective, **program["kwargs"])
        if result.status != 0:
            # Presolve now and then misjudges a system; an infeasibility (or no
            # verdict) is taken only once a run without it agrees.
            result = linprog(objective, options={"presolve": False}, **program["kwargs"])
        # Only a proof of infeasibility counts against the system (status 2).
        return result.status != 2

    def _hide_beside(self, failing, hidden) -> bool:
        """Hide a neighbour of every unprotected cell; whether any was hidden."""
        added = False
        for cell in failing:
            pick = self._neighbour(cell, hidden)
            if pick is not None:
                self.forced.add(pick)
                hidden[pick] = COMPLEMENT
                added = True
        return added

    def _complement(self, min_cell: int) -> Dict[int, str]:
        hidden: Dict[int, str] = {}
        for index, value in enumerate(self.values):
            if not self.fixed[index] and 0 < value < min_cell:
                hidden[index] = PRIMARY
        for index in self.forced:
            hidden.setdefault(index, COMPLEMENT)
        for index in self.latent_cells:
            hidden[index] = LATENT
        relations = [(t, p) for t, p in self.constraints + self.links if not self._has_latent(t, p)]
        changed = True
        while changed:
            changed = False
            for total, parts in relations:
                pick = self._repair(total, parts, hidden, min_cell)
                if pick is not None:
                    hidden[pick] = COMPLEMENT
                    changed = True
        return hidden

    def _repair(self, total, parts, hidden, min_cell) -> Optional[int]:
        shown = [p for p in parts if p not in hidden]
        candidates = [p for p in shown if not self.fixed[p] and self.values[p] > 0]
        if total in hidden:
            if shown and len(shown) == len(parts):
                return _smallest(candidates, self.values)
            return None
        masked = [p for p in parts if p in hidden]
        if not masked:
            return None
        if len(masked) >= 2 and sum(self.values[p] for p in masked) >= min_cell:
            return None
        pick = _smallest(candidates, self.values)
        if pick is None and not self.fixed[total] and self.values[total] > 0:
            return total
        return pick

    def _neighbour(self, cell: int, hidden) -> Optional[int]:
        """The smallest shown, non-fixed, positive cell sharing a constraint with ``cell``.

        Failing that, one sharing a constraint with any hidden cell of the
        same component; None when only fixed cells are left.
        """
        def around(targets):
            found = set()
            for total, parts in self.constraints + self.links + self.subsets:
                members = (total,) + tuple(parts)
                if any(t in members for t in targets):
                    found.update(members)
            return [c for c in sorted(found)
                    if c not in hidden and not self.fixed[c] and self.values[c] > 0]

        pick = _smallest(around({cell}), self.values)
        if pick is None:
            component = next((c for c in self._components() if cell in c[0]), (set(), [], []))[0]
            pick = _smallest(around({c for c in component if c in hidden}), self.values)
        return pick

    # -- audit -------------------------------------------------------------------

    def _numeric(self) -> List[Tuple[int, Tuple[int, ...]]]:
        """Constraints a reader can compute with: every sum, and links of equal values."""
        equal = [(t, p) for t, p in self.links if self.values[t] == self.values[p[0]]]
        return self.constraints + equal

    def _components(self):
        """``(cells, equalities, inequalities)`` per connected component.

        Fixed cells are known constants: they join nothing, so two tables that
        share only a fixed total are audited apart.
        """
        parent = list(range(len(self.values)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        numeric = self._numeric()
        relations = [(r, False) for r in numeric] + [(r, True) for r in self.subsets]
        for (total, parts), _ in relations:
            free = [m for m in (total,) + tuple(parts) if not self.fixed[m]]
            for member in free[1:]:
                a, b = find(free[0]), find(member)
                if a != b:
                    parent[a] = b
        groups: Dict[int, Tuple[Set[int], List, List]] = {}
        for index in range(len(self.values)):
            if not self.fixed[index]:
                groups.setdefault(find(index), (set(), [], []))[0].add(index)
        for (total, parts), inequality in relations:
            free = [m for m in (total,) + tuple(parts) if not self.fixed[m]]
            if free:
                groups[find(free[0])][2 if inequality else 1].append((total, parts))
        return list(groups.values())

    def _program(self, component, hidden):
        """The linear system of one component: the reader's unknowns and relations."""
        import numpy as np
        from scipy.sparse import csr_matrix

        members, equalities, inequalities = component
        unknown = sorted(c for c in members if c in hidden)
        position = {c: k for k, c in enumerate(unknown)}

        def rows(relations):
            data, cols, starts, b = [], [], [0], []
            for total, parts in relations:
                row: Dict[int, float] = {}
                constant = 0.0
                for member, sign in [(total, -1.0)] + [(p, 1.0) for p in parts]:
                    if member in position:
                        row[position[member]] = row.get(position[member], 0.0) + sign
                    else:
                        constant -= sign * self.values[member]
                row = {k: v for k, v in row.items() if v != 0.0}
                if row:
                    cols.extend(row)
                    data.extend(row.values())
                    starts.append(len(cols))
                    b.append(constant)
            if not b:
                return None, None
            return (csr_matrix((data, cols, starts), shape=(len(b), len(unknown))),
                    np.array(b))

        a_eq, b_eq = rows(equalities)
        a_ub, b_ub = rows(inequalities)
        # A finite cap above any count: HiGHS' presolve can abort the process on
        # an unbounded maximum (scipy 1.13), and a huge cap makes it lose its
        # way numerically. Every true value, and so every lower end, is below
        # the largest count, so a maximum at the cap is far enough above both
        # to be "unbounded" for every purpose here.
        known = [abs(v) for v in self.values if v == v]
        cap = min(_CAP, 2.0 * max(known, default=0.0) + 1000.0)
        # A hidden shown-table cell is >= 1 (a zero is shown); a latent one >= 0.
        box = [(0 if c in self.latent_cells else 1, cap) for c in unknown]
        return {"position": position, "box": box, "cap": cap,
                "empty": a_eq is None and a_ub is None,
                "kwargs": dict(A_eq=a_eq, b_eq=b_eq, A_ub=a_ub, b_ub=b_ub, bounds=box,
                               method="highs")}

    def _range(self, program, cell: int, need_low: bool = True) -> Tuple[float, float]:
        import numpy as np
        from scipy.optimize import linprog

        floor = 0.0 if cell in self.latent_cells else 1.0
        if program["empty"]:
            return floor, math.inf
        objective = np.zeros(len(program["box"]))
        objective[program["position"][cell]] = 1.0

        def optimum(sign):
            result = linprog(sign * objective, **program["kwargs"])
            if result.status != 0:
                # HiGHS now and then stops without a verdict; without presolve
                # it gets there. No verdict at all is NaN, which audit counts as
                # unprotected (a floor of 0 would read as safe for a latent cell).
                result = linprog(sign * objective, options={"presolve": False},
                                 **program["kwargs"])
            return sign * result.fun if result.status == 0 else math.nan

        hi = optimum(-1.0)
        lo = optimum(1.0) if need_low else floor
        return float(lo), (math.inf if hi >= program["cap"] - 1 else float(hi))

    def bounds(self, cell: int, hidden, component=None) -> Tuple[float, float]:
        """Feasible ``[lo, hi]`` of a hidden cell under the attacker model."""
        if component is None:
            component = next(c for c in self._components() if cell in c[0])
        return self._range(self._program(component, hidden), cell)

    def audit(self, min_cell: int, hidden, components=None, memo=None,
              first: bool = False, early: Optional[Set[int]] = None) -> List[int]:
        """Small hidden cells whose feasible range does not protect them.

        ``memo`` keeps each component's result for the hidden cells it had,
        so a fixed-point round re-solves only the components that changed.
        ``first``: stop at the first unprotected cell (a yes/no answer; a
        memo used with it must be used with it only); ``early``: cells to
        check before the others.
        """
        failing = []
        for component in components if components is not None else self._components():
            members = component[0]
            mine = frozenset(c for c in members if c in hidden)
            targets = sorted(c for c in mine
                             if 0 < self.values[c] < min_cell or c in self.latent_audited)
            if not targets:
                continue
            key = (min(members), mine)
            if memo is not None and key in memo:
                failing.extend(memo[key])
                continue
            # Hiding more cells only widens every range (an old reading stays a
            # reading, the newly hidden cells at their values), so a cell found
            # protected under fewer hidden cells stays protected.
            safe = memo.setdefault("protected", {}) if memo is not None else {}
            targets = [c for c in targets if not (c in safe and safe[c] <= mine)]
            if early:
                targets.sort(key=lambda c: c not in early)
            found = []
            program = self._program(component, hidden) if targets else None
            for cell in targets:
                lo, hi = self._range(program, cell)
                if math.isnan(lo) or math.isnan(hi):
                    found.append(cell)
                elif cell in self.latent_cells and (hi < 1 - 1e-6 or lo > min_cell - 1 + 1e-6):
                    # No value from 1 to min_cell - 1 is possible, but only with
                    # exactly these cells hidden: more hidden cells widen the
                    # range into that interval, so this verdict is not kept in
                    # ``safe`` (memo[key] keeps it for this hidden set only).
                    pass
                elif hi < min_cell - 1e-6 or hi - lo < min_cell - 1 - 1e-6:
                    found.append(cell)
                else:
                    safe[cell] = mine
                if first and found:
                    break
            if memo is not None:
                memo[key] = found
            failing.extend(found)
            if first and failing:
                break
        return failing

    def violations(self, min_cell: int, hidden: Dict[int, str]) -> List[Tuple[int, Tuple[int, ...]]]:
        """Constraints that still give a hidden count (or sum) of 1 to ``min_cell - 1`` away."""
        bad = []
        for total, parts in self._numeric():
            if self._has_latent(total, parts):
                continue
            masked = [p for p in parts if p in hidden]
            if total in hidden:
                if not masked:
                    derived = sum(self.values[p] for p in parts)
                    if 0 < derived < min_cell:
                        bad.append((total, parts))
            elif masked:
                derived = sum(self.values[p] for p in masked)
                if 0 < derived < min_cell:
                    bad.append((total, parts))
        return bad


def _smallest(candidates, values) -> Optional[int]:
    if not candidates:
        return None
    return min(candidates, key=lambda p: (values[p], p))


def _number(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number
