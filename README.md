# Pressure screen

Which issuers come under pressure if inflation stays high and growth slows?

This is a research screen. It ranks issuers by exposure. It gives no trade, size, hedge or stop.
The design is in `docs/design.md`. Thresholds are in `config/screen.yaml`, each with its source.

## Run it in four steps

1. Copy `config/sources.example.yaml` to `config/sources.yaml`. Map each table to your data.
2. Check the data: `python -m pressure_screen doctor --sources config/sources.yaml`
3. Run the screen: `python -m pressure_screen run --sources config/sources.yaml --as-of 2026-11-20 --out out/run`
4. Test it on the past: `python -m pressure_screen backtest --sources config/sources.yaml --formation 2021-12-31 --end 2023-12-31 --peak 2022-10-14 --out out/bt`

Fix every ERROR row in step 2 first. Steps 3 and 4 refuse to run until then.

## Try it with invented data first

    python -m pressure_screen synth --out demo
    python -m pressure_screen backtest --sources demo/sources.yaml --formation 2021-12-31 --end 2023-12-31 --peak 2022-10-14 --out demo/bt

The invented data has a planted signal, so the backtest passes. Add `--no-signal` and it must not pass.

## How it scores

- **Gate 0, data.** An issuer needs 8 of the 11 variables. Otherwise it is "unscored". Unscored means unknown, not safe.
- **Gate 1, rate channel.** Refinancing need above 1.0, or floating share above 50%, or interest cover below 2.0 after refinancing.
- **Gate 2, earnings channel.** Worst third of its sector on the margin group or the demand group.
- **Already priced.** Bonds under 80 cents or a CCC-implied spread. These go on a separate list.
- **Rank.** Each variable becomes a percentile inside the rating tier and sector. 100 is worst. `F` is the equal-weight mean of the three fundamental groups. `M` is the market group. A high `F` with a low `M` is pressure that prices do not yet show.

## The backtest fails the screen if any one of these fails

1. The top fifth by `F` has less than 1.5 times the downgrade and default rate of the bottom fifth.
2. `F` adds nothing inside thirds of the starting spread.
3. The result disappears when one sector is dropped.
4. The gates pass more than 40% of scored issuers.
5. High `F` with low `M` does not widen more than high `F` with high `M`.

A test the data cannot run is "NOT TESTABLE". The overall verdict is then "INCOMPLETE", never "PASS".

## Rules the code keeps

- It uses only data dated on or before the as-of date. Filings count from `filed_date`, not `period_end`.
- A missing input gives a missing variable. It never gives a safe value.
- An empty table is reported as EMPTY. A table whose issuer ids match nothing is an ERROR.
- Filings older than 200 days are stale. Their variables are missing.

## Tests

    python -m pytest tests -q

## Known limits

- Equal weights are a choice. Nothing is fitted.
- "Convention" thresholds in `config/screen.yaml` come from analyst practice, not from a book.
- The `market_stress` implied rating uses the nearest rating-tier spread. It is coarse.
- A window with no recession tests the demand leg weakly.
- The WRDS query templates are untested. Only the marketdb bond query ran on real data.
