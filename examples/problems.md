# Problems to try

Paste any of these into the `optimist` REPL. They span the problem classes the
agent team handles: LP, MILP, scheduling (CP-SAT), assignment, routing, and
pure constraint satisfaction.

## Production mix (LP)

> A furniture shop makes desks and chairs. A desk needs 4 hours of carpentry
> and 2 hours of finishing; a chair needs 3 hours of carpentry and 1 hour of
> finishing. Each week there are 240 carpentry hours and 100 finishing hours.
> A desk earns $70 profit, a chair $50. What weekly production plan maximizes
> profit?

## Knapsack (MILP)

> I can carry 15 kg. Items: tent 5 kg (value 9), stove 3 kg (value 6), camera
> 2 kg (value 7), food 6 kg (value 8), books 4 kg (value 3). Which items should
> I take to maximize value?

## Job-shop scheduling (CP-SAT) — the problem this repo was born for

> Three jobs must each pass through machines M1, M2, M3 in a job-specific
> order, one operation at a time, machines handle one job at a time:
> Job A: M1 (3h) → M2 (2h) → M3 (2h)
> Job B: M1 (2h) → M3 (1h) → M2 (4h)
> Job C: M2 (4h) → M3 (3h) → M1 (4h)
> Minimize the makespan and give me the schedule.

## Assignment

> Four reviewers must each take exactly one paper. Estimated review quality
> (0-10): Alice: P1=9, P2=4, P3=6, P4=5; Bob: P1=7, P2=8, P3=5, P4=6;
> Carol: P1=3, P2=6, P3=9, P4=7; Dan: P1=6, P2=5, P3=4, P4=9.
> Maximize total quality.

## Shift scheduling (feasibility / CSP)

> A clinic needs 2 nurses on Mon/Wed/Fri and 3 on Tue/Thu. Nurses A-F each
> work at most 3 days a week; nobody works two consecutive days; A and B
> can't work the same day. Find a valid week or tell me it's impossible.

## Diet problem (classic LP, with a twist)

> Minimize the cost of a diet from: oats ($0.30/serving: 110 cal, 4 g protein),
> chicken ($2.00: 205 cal, 32 g protein), eggs ($0.50: 160 cal, 13 g protein),
> milk ($0.90: 160 cal, 8 g protein). I need at least 2000 calories and 90 g
> protein per day, at most 4 servings of any single food. What if I also
> refuse to eat more than 2 eggs?

## Infeasibility diagnosis

> Schedule 10 one-hour meetings for the same 3 people between 9:00 and 12:00
> on one day. (This is impossible — the point is to see the agents prove it
> and explain which constraints collide.)

## With an attachment

```
/attach tender_bids.pdf
Assign each construction lot to exactly one contractor minimizing total cost,
no contractor gets more than two lots. The bid table is in the PDF.
```
