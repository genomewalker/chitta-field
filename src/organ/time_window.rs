// Time-window parser: deterministic natural-language temporal phrases → ts_ms
// window. No LLM, no allocation beyond the stripped query. The window GATES
// recall candidates; semantic similarity still ranks (recency-sort pollutes —
// see the stored temporal-recall corrections). No match → None → recall
// behavior is byte-identical to a query without a temporal phrase.
//
// Semantics (owner-approved):
//   "last week"/"last month"/"last year"  → previous CALENDAR period
//   "past week"/"last 7 days"/"last N h"  → ROLLING window ending now
//   Weeks start Monday. All calendar math in the caller's local wall time
//   (tz_offset_min), converted back to UTC ms.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub struct TimeWindow {
    pub from_ms: i64,
    pub to_ms: i64,
    /// Query with the temporal phrase removed (for embedding).
    pub stripped: String,
}

const MS_HOUR: i64 = 3_600_000;
const MS_DAY: i64 = 86_400_000;

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = (mp + 2) % 12 + 1;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 0 = Monday .. 6 = Sunday.
fn weekday_from_days(z: i64) -> i64 {
    (z + 3).rem_euclid(7)
}

fn month_len(y: i64, m: i64) -> i64 {
    days_from_civil(y + (m == 12) as i64, m % 12 + 1, 1) - days_from_civil(y, m, 1)
}

fn month_num(name: &str) -> Option<i64> {
    const MONTHS: [&str; 12] = [
        "january", "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december",
    ];
    MONTHS.iter().position(|m| *m == name).map(|i| i as i64 + 1)
}

/// Local-wall-time context derived from now.
struct Ctx {
    now_local_ms: i64,
    today_days: i64, // local civil days since epoch
    tz_ms: i64,
}

impl Ctx {
    fn day_start_ms(&self, days: i64) -> i64 {
        days * MS_DAY // local midnight, in local ms
    }
    fn month_start_days(&self, y: i64, m: i64) -> i64 {
        days_from_civil(y, m, 1)
    }
    /// Convert a local-ms instant back to UTC ms.
    fn to_utc(&self, local_ms: i64) -> i64 {
        local_ms - self.tz_ms
    }
    fn ymd(&self) -> (i64, i64, i64) {
        civil_from_days(self.today_days)
    }
}

struct Rules {
    last_n: Regex,       // last/past N hours|days|weeks|months
    past_period: Regex,  // past week|month|year (rolling)
    calendar: Regex,     // this|last week|month|year (calendar)
    yesterday: Regex,
    today: Regex,
    between: Regex,
    since: Regex,        // since|after <date>
    before: Regex,
    in_month: Regex,     // in June [2026] | in 2026-06 | in 2026
    on_date: Regex,      // on 2026-06-05
}

fn rules() -> &'static Rules {
    static R: OnceLock<Rules> = OnceLock::new();
    const MON: &str = "january|february|march|april|may|june|july|august|september|october|november|december";
    let date = format!(r"(\d{{4}}-\d{{2}}-\d{{2}}|\d{{4}}-\d{{2}}|(?:{MON})(?:\s+\d{{4}})?|\d{{4}})");
    R.get_or_init(|| Rules {
        last_n: Regex::new(r"(?i)\b(?:last|past)\s+(\d{1,3})\s+(hour|day|week|month)s?\b").unwrap(),
        past_period: Regex::new(r"(?i)\bpast\s+(week|month|year)\b").unwrap(),
        calendar: Regex::new(r"(?i)\b(this|last)\s+(week|month|year)\b").unwrap(),
        yesterday: Regex::new(r"(?i)\byesterday\b").unwrap(),
        today: Regex::new(r"(?i)\btoday\b").unwrap(),
        between: Regex::new(&format!(r"(?i)\bbetween\s+{date}\s+and\s+{date}\b")).unwrap(),
        since: Regex::new(&format!(r"(?i)\b(?:since|after)\s+{date}\b")).unwrap(),
        before: Regex::new(&format!(r"(?i)\bbefore\s+{date}\b")).unwrap(),
        in_month: Regex::new(&format!(r"(?i)\bin\s+{date}\b")).unwrap(),
        on_date: Regex::new(r"(?i)\bon\s+(\d{4}-\d{2}-\d{2})\b").unwrap(),
    })
}

/// Parse a date phrase (captured by the `date` sub-pattern) into the local-ms
/// range it denotes: (start, end). Month names resolve to the most recent
/// occurrence that is not in the future.
fn parse_date_phrase(s: &str, ctx: &Ctx) -> Option<(i64, i64)> {
    let s = s.trim().to_lowercase();
    let (cy, cm, _) = ctx.ymd();
    if let Some((ym, d)) = s
        .split_once('-')
        .and_then(|(y, rest)| rest.split_once('-').map(|(m, d)| ((y.to_string(), m.to_string()), d.to_string())))
    {
        // yyyy-mm-dd
        let (y, m, d) = (ym.0.parse().ok()?, ym.1.parse().ok()?, d.parse().ok()?);
        let start = ctx.day_start_ms(days_from_civil(y, m, d));
        return Some((start, start + MS_DAY));
    }
    if let Some((y, m)) = s.split_once('-') {
        // yyyy-mm
        let (y, m): (i64, i64) = (y.parse().ok()?, m.parse().ok()?);
        if !(1..=12).contains(&m) {
            return None;
        }
        let start = ctx.day_start_ms(ctx.month_start_days(y, m));
        return Some((start, start + month_len(y, m) * MS_DAY));
    }
    if let Ok(y) = s.parse::<i64>() {
        // yyyy — reject implausible years so "in 45 minutes" never matches
        if !(1990..=2100).contains(&y) {
            return None;
        }
        let start = ctx.day_start_ms(days_from_civil(y, 1, 1));
        return Some((start, ctx.day_start_ms(days_from_civil(y + 1, 1, 1))));
    }
    // month name [year]
    let mut parts = s.split_whitespace();
    let m = month_num(parts.next()?)?;
    let y = match parts.next() {
        Some(y) => y.parse().ok()?,
        None => {
            // most recent non-future occurrence
            if m > cm { cy - 1 } else { cy }
        }
    };
    let start = ctx.day_start_ms(ctx.month_start_days(y, m));
    Some((start, start + month_len(y, m) * MS_DAY))
}

/// Parse a temporal phrase out of `query`. `now_ms` is UTC epoch ms;
/// `tz_offset_min` is the caller's local UTC offset in minutes (CEST = 120).
/// Returns None when no phrase matches — the no-match guarantee.
pub fn parse_time_window(query: &str, now_ms: i64, tz_offset_min: i32) -> Option<TimeWindow> {
    let tz_ms = tz_offset_min as i64 * 60_000;
    let now_local_ms = now_ms + tz_ms;
    let ctx = Ctx { now_local_ms, today_days: now_local_ms.div_euclid(MS_DAY), tz_ms };
    let r = rules();

    // (match_range, local (from, to)) — ordered most-specific first; first hit wins.
    let hit: Option<(std::ops::Range<usize>, (i64, i64))> = None
        .or_else(|| {
            r.between.captures(query).and_then(|c| {
                let a = parse_date_phrase(c.get(1)?.as_str(), &ctx)?;
                let b = parse_date_phrase(c.get(2)?.as_str(), &ctx)?;
                Some((c.get(0)?.range(), (a.0.min(b.0), a.1.max(b.1))))
            })
        })
        .or_else(|| {
            r.since.captures(query).and_then(|c| {
                let d = parse_date_phrase(c.get(1)?.as_str(), &ctx)?;
                Some((c.get(0)?.range(), (d.0, ctx.now_local_ms)))
            })
        })
        .or_else(|| {
            r.before.captures(query).and_then(|c| {
                let d = parse_date_phrase(c.get(1)?.as_str(), &ctx)?;
                Some((c.get(0)?.range(), (0, d.0)))
            })
        })
        .or_else(|| {
            r.on_date.captures(query).and_then(|c| {
                let d = parse_date_phrase(c.get(1)?.as_str(), &ctx)?;
                Some((c.get(0)?.range(), d))
            })
        })
        .or_else(|| {
            r.last_n.captures(query).and_then(|c| {
                let n: i64 = c.get(1)?.as_str().parse().ok()?;
                let unit_ms = match c.get(2)?.as_str().to_lowercase().as_str() {
                    "hour" => MS_HOUR,
                    "day" => MS_DAY,
                    "week" => 7 * MS_DAY,
                    _ => 30 * MS_DAY, // month, rolling approximation
                };
                Some((c.get(0)?.range(), (ctx.now_local_ms - n * unit_ms, ctx.now_local_ms)))
            })
        })
        .or_else(|| {
            r.past_period.captures(query).and_then(|c| {
                let span = match c.get(1)?.as_str().to_lowercase().as_str() {
                    "week" => 7 * MS_DAY,
                    "month" => 30 * MS_DAY,
                    _ => 365 * MS_DAY,
                };
                Some((c.get(0)?.range(), (ctx.now_local_ms - span, ctx.now_local_ms)))
            })
        })
        .or_else(|| {
            r.calendar.captures(query).and_then(|c| {
                let which = c.get(1)?.as_str().to_lowercase();
                let (cy, cm, _) = ctx.ymd();
                let win = match c.get(2)?.as_str().to_lowercase().as_str() {
                    "week" => {
                        let monday = ctx.today_days - weekday_from_days(ctx.today_days);
                        if which == "this" {
                            (ctx.day_start_ms(monday), ctx.now_local_ms)
                        } else {
                            (ctx.day_start_ms(monday - 7), ctx.day_start_ms(monday))
                        }
                    }
                    "month" => {
                        if which == "this" {
                            (ctx.day_start_ms(ctx.month_start_days(cy, cm)), ctx.now_local_ms)
                        } else {
                            let (py, pm) = if cm == 1 { (cy - 1, 12) } else { (cy, cm - 1) };
                            let s = ctx.month_start_days(py, pm);
                            (ctx.day_start_ms(s), ctx.day_start_ms(s + month_len(py, pm)))
                        }
                    }
                    _ => {
                        if which == "this" {
                            (ctx.day_start_ms(days_from_civil(cy, 1, 1)), ctx.now_local_ms)
                        } else {
                            (
                                ctx.day_start_ms(days_from_civil(cy - 1, 1, 1)),
                                ctx.day_start_ms(days_from_civil(cy, 1, 1)),
                            )
                        }
                    }
                };
                Some((c.get(0)?.range(), win))
            })
        })
        .or_else(|| {
            r.yesterday.find(query).map(|m| {
                (m.range(), (ctx.day_start_ms(ctx.today_days - 1), ctx.day_start_ms(ctx.today_days)))
            })
        })
        .or_else(|| {
            r.today
                .find(query)
                .map(|m| (m.range(), (ctx.day_start_ms(ctx.today_days), ctx.now_local_ms)))
        })
        .or_else(|| {
            r.in_month.captures(query).and_then(|c| {
                let d = parse_date_phrase(c.get(1)?.as_str(), &ctx)?;
                Some((c.get(0)?.range(), d))
            })
        });

    let (range, (from_local, to_local)) = hit?;
    let mut stripped = String::with_capacity(query.len());
    stripped.push_str(&query[..range.start]);
    stripped.push(' ');
    stripped.push_str(&query[range.end..]);
    let stripped = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(TimeWindow {
        from_ms: ctx.to_utc(from_local),
        to_ms: ctx.to_utc(to_local),
        stripped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-07-06 is a Monday. now = 2026-07-06 14:00 UTC.
    const NOW: i64 = 1_783_346_400_000;
    const TZ: i32 = 120; // CEST

    fn win(q: &str) -> Option<TimeWindow> {
        parse_time_window(q, NOW, TZ)
    }

    fn day_utc(y: i64, m: i64, d: i64) -> i64 {
        days_from_civil(y, m, d) * MS_DAY - (TZ as i64) * 60_000
    }

    #[test]
    fn no_match_is_none() {
        assert!(win("how does the span lane dedup work").is_none());
        assert!(win("the last word on this").is_none()); // no unit → no match
        assert!(win("in 45 minutes").is_none()); // implausible year guard
    }

    #[test]
    fn calendar_last_week_is_previous_mon_to_mon() {
        let w = win("what did we do last week on the span lane").unwrap();
        assert_eq!(w.from_ms, day_utc(2026, 6, 29));
        assert_eq!(w.to_ms, day_utc(2026, 7, 6));
        assert_eq!(w.stripped, "what did we do on the span lane");
    }

    #[test]
    fn rolling_last_n_days() {
        let w = win("failures in the last 3 days").unwrap();
        assert_eq!(w.to_ms, NOW);
        assert_eq!(w.from_ms, NOW - 3 * MS_DAY);
        assert_eq!(w.stripped, "failures in the");
    }

    #[test]
    fn past_week_is_rolling() {
        let w = win("past week reranker changes").unwrap();
        assert_eq!((w.from_ms, w.to_ms), (NOW - 7 * MS_DAY, NOW));
    }

    #[test]
    fn yesterday_today() {
        let y = win("what broke yesterday").unwrap();
        assert_eq!((y.from_ms, y.to_ms), (day_utc(2026, 7, 5), day_utc(2026, 7, 6)));
        let t = win("what did we ship today").unwrap();
        assert_eq!((t.from_ms, t.to_ms), (day_utc(2026, 7, 6), NOW));
    }

    #[test]
    fn in_month_resolves_most_recent() {
        let w = win("what changed in june with the reranker").unwrap();
        assert_eq!(w.from_ms, day_utc(2026, 6, 1));
        assert_eq!(w.to_ms, day_utc(2026, 7, 1));
        assert_eq!(w.stripped, "what changed with the reranker");
        // future month → previous year
        let w = win("in december deploys").unwrap();
        assert_eq!(w.from_ms, day_utc(2025, 12, 1));
    }

    #[test]
    fn since_before_between_on() {
        let s = win("since 2026-05-15 what merged").unwrap();
        assert_eq!((s.from_ms, s.to_ms), (day_utc(2026, 5, 15), NOW));
        let b = win("before march 2026 decisions").unwrap();
        assert_eq!((b.from_ms, b.to_ms), (0 - (TZ as i64) * 60_000, day_utc(2026, 3, 1)));
        let r = win("between 2026-04 and 2026-06 refactors").unwrap();
        assert_eq!((r.from_ms, r.to_ms), (day_utc(2026, 4, 1), day_utc(2026, 7, 1)));
        let o = win("what happened on 2026-06-15").unwrap();
        assert_eq!((o.from_ms, o.to_ms), (day_utc(2026, 6, 15), day_utc(2026, 6, 16)));
    }

    #[test]
    fn last_month_is_calendar() {
        let w = win("last month migration work").unwrap();
        assert_eq!(w.from_ms, day_utc(2026, 6, 1));
        assert_eq!(w.to_ms, day_utc(2026, 7, 1));
    }
}
