use std::collections::HashSet;

use vecdb::{Config, Filter, Index, Metric, Payload, Value};

// 200 docs: lang cycles ar/en/fr, year = 2000 + i % 10, rating is a float (not indexed)
fn docs() -> Index {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 1000;
    let idx = Index::new(cfg).unwrap();
    for i in 0..200u64 {
        let mut p = Payload::new();
        p.insert("lang".into(), Value::Str(["ar", "en", "fr"][(i % 3) as usize].into()));
        p.insert("year".into(), Value::Int(2000 + (i % 10) as i64));
        p.insert("rating".into(), Value::Float(i as f64 / 10.0));
        idx.insert(i, &[i as f32, 1.0], p).unwrap();
    }
    idx
}

fn all_pages(idx: &Index, f: Option<&Filter>, limit: usize) -> Vec<u64> {
    let (mut ids, mut cur) = (vec![], 0);
    loop {
        let page = idx.scroll(f, cur, limit, false);
        assert!(page.items.len() <= limit);
        ids.extend(page.items.iter().map(|r| r.id));
        match page.next {
            Some(n) => cur = n,
            None => return ids,
        }
    }
}

#[test]
fn scroll_visits_everything_once() {
    let idx = docs();
    idx.delete(5).unwrap();
    idx.delete(150).unwrap();
    let ids = all_pages(&idx, None, 17);
    assert_eq!(ids.len(), 198);
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), 198);
    assert!(!ids.contains(&5) && !ids.contains(&150));
    assert!(ids.windows(2).all(|w| w[0] < w[1]));

    // an exact multiple of the page size must not produce a trailing empty page
    let page = idx.scroll(None, 0, 198, false);
    assert_eq!(page.items.len(), 198);
    assert!(page.next.is_none());
}

#[test]
fn scroll_with_filters_and_vectors() {
    let idx = docs();
    let ar = Filter::eq("lang", "ar");
    let ids = all_pages(&idx, Some(&ar), 7);
    assert_eq!(ids, (0..200).filter(|i| i % 3 == 0).collect::<Vec<_>>());

    // range filters can't use the inverted index, they take the scanning path
    let late = Filter::range("rating", Some(15.0), None);
    assert_eq!(all_pages(&idx, Some(&late), 9), (150..200).collect::<Vec<_>>());

    // mixed: indexed part narrows, the range part is verified per row
    let both = Filter::And(vec![ar, late]);
    assert_eq!(all_pages(&idx, Some(&both), 4), (150..200).filter(|i| i % 3 == 0).collect::<Vec<_>>());

    let page = idx.scroll(None, 10, 2, true);
    assert_eq!(page.items[0].vector, vec![10.0, 1.0]);
    assert_eq!(page.next, Some(12));
    assert!(idx.scroll(None, 5000, 10, false).items.is_empty());
}

#[test]
fn count_agrees_with_scroll() {
    let idx = docs();
    idx.delete(3).unwrap();
    assert_eq!(idx.count(None), 199);
    let filters = [
        Filter::eq("lang", "ar"),
        Filter::range("rating", Some(5.0), Some(9.95)),
        Filter::Not(Box::new(Filter::eq("lang", "en"))),
        Filter::Or(vec![Filter::eq("year", 2001), Filter::eq("year", 2002)]),
    ];
    for f in &filters {
        assert_eq!(idx.count(Some(f)), all_pages(&idx, Some(f), 50).len(), "{f:?}");
    }
    assert_eq!(idx.count(Some(&Filter::eq("lang", "zz"))), 0);
}

#[test]
fn facets_count_values() {
    let idx = docs();
    let langs = idx.facets("lang", None, 10);
    assert_eq!(langs.len(), 3);
    assert_eq!(langs[0], (Value::Str("ar".into()), 67)); // 0, 3 .. 198
    assert_eq!(langs[1], (Value::Str("en".into()), 67));
    assert_eq!(langs[2], (Value::Str("fr".into()), 66));

    assert_eq!(idx.facets("lang", None, 2).len(), 2);

    let late = Filter::range("rating", Some(10.0), None); // ids 100..200
    let years = idx.facets("year", Some(&late), 20);
    assert_eq!(years.len(), 10);
    assert!(years.iter().all(|(_, n)| *n == 10));

    idx.delete(0).unwrap();
    assert_eq!(idx.facets("lang", None, 1)[0], (Value::Str("en".into()), 67));

    assert!(idx.facets("rating", None, 5).is_empty());
    assert!(idx.facets("missing", None, 5).is_empty());
}
