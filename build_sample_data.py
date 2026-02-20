# -*- coding: utf-8 -*-
import json, pathlib

PATCH = "Proposes extending the Linux kernel's Transparent Huge Pages (THP) subsystem to support 1GB huge pages on x86-64, targeting terabyte-scale NUMA machines where the current 2MB ceiling creates excessive page table overhead and TLB pressure. The author benchmarks show a 12-18% reduction in kernel memory management overhead on 4TB systems."

DH_B = "On Thu, 13 Feb 2026, Usama Arif wrote:\n> We propose adding 1GB THP support...\n\nI'm concerned about compaction. 1GB pages are essentially impossible to\nmigrate once faulted in. What happens under memory pressure on a machine\nthat isn't actually running a terabyte-scale workload?\n\nAlso, the benchmark numbers look good for your target workload but I'd\nlove to see fragmentation numbers after a few hours of mixed workload.\n\nDavid"

JW_B = "On Thu, 13 Feb 2026, David Hildenbrand wrote:\n> I'm concerned about compaction...\n\nFair point, but we already have similar issues with 2MB pages at the\nextreme end. The migration scanner just gives up eventually.\n\n> Also, the benchmark numbers look good...\n\nAgreed on fragmentation data. However, I wonder if a per-cgroup\nkhugepaged knob would make the tradeoff explicit: admins who know\nthey're running on terabyte-scale dedicated hardware can opt in,\nothers are unaffected.\n\nJohannes"

ZY1_B = "On Fri, 14 Feb 2026, Johannes Weiner wrote:\n> I wonder if a per-cgroup khugepaged knob would make the tradeoff explicit\n\n+1 for the knob idea. I'd suggest exposing it via sysfs under:\n  /sys/kernel/mm/transparent_hugepage/enabled_sizes\n\nwhich already exists for the 2MB case. We could add a '1g' token.\n\nAlso worth considering: using CMA at boot to pre-reserve 1GB-aligned\nregions. This avoids the runtime allocation headache entirely for\ndedicated workloads.\n\nZi Yan"

ZY2_B = 'On Thu, 13 Feb 2026, David Hildenbrand wrote:\n> 1GB pages are essentially impossible to migrate once faulted in.\n\nTrue, but MADV_COLLAPSE and MADV_HUGEPAGE already give userspace\ncontrol for 2MB. We could add MADV_NO_1GHUGEPAGE for VMAs where\nlatency-sensitive code needs to opt out.\n\nZi Yan'

RVR_B = "Usama, all —\n\nI want to flag a NUMA interaction issue that I don't see addressed in\nthe proposal. On a 4-socket machine with 1TB per node, a 1GB THP that\nspans a NUMA boundary would have half its memory remote. The NUMA\nfault handler would never be able to fix this without splitting the\nhuge page, which we can't do for 1GB pages.\n\nI think we need: 1GB THP allocation must be NUMA-local-only, with\nstrict fallback to 2MB when cross-node placement would occur.\n\nRik"

DH2_B = "On Fri, 14 Feb 2026, Rik van Riel wrote:\n> I think we need: 1GB THP allocation must be NUMA-local-only\n\nStrongly agree. And I'd go further: this should be a hard kernel\ninvariant enforced at allocation time. Not a documentation caveat,\nnot a sysctl knob — if the physical 1GB region would span NUMA nodes,\nwe must not allocate it as a 1GB THP, period.\n\nThe implication is that 1GB THP is effectively a no-op on most real\nNUMA servers unless the NUMA topology happens to align at 1GB\ngranularity, which is rare.\n\nDavid"

MW_B = "On Fri, 14 Feb 2026, Zi Yan wrote:\n> Also worth considering: using CMA at boot to pre-reserve 1GB-aligned regions.\n\nI don't love the CMA approach. You're permanently carving memory out\nof the buddy allocator at boot, which creates a fragmentation cliff\nfor everyone else on the machine.\n\nWe already have the large folio infrastructure. Why not use it\nopportunistically: only promote to 1GB when a naturally 1GB-aligned\nregion is available in the buddy allocator? No reservation needed,\nno cliff.\n\nWlcx"

DH_S = 'Raised concerns about the impact on memory compaction and migration, noting that 1GB pages are essentially impossible to migrate under memory pressure. Questioned whether the benefit on terabyte-scale machines justifies the compaction regression for all other workloads. Requested benchmark data covering fragmentation scenarios.'

JW_S = "Partially agreed with David's migration concern but pointed out that the kernel already has similar issues with 2MB pages under extreme memory pressure. Suggested that a khugepaged policy knob could allow admins to opt-in per-cgroup, making the tradeoff explicit rather than global."

ZY1_S = "Supported Johannes's cgroup knob idea and proposed a concrete sysfs interface under /sys/kernel/mm/transparent_hugepage/enabled_sizes. Also noted that the CMA (Contiguous Memory Allocator) could be leveraged at boot time to reserve a pool of 1GB-aligned physical regions, reducing runtime allocation failures."

ZY2_S = "Directly addressed David's compaction concern by pointing to existing kernel mechanisms: MADV_COLLAPSE and madvise(MADV_HUGEPAGE) already give userspace an escape valve. Proposed adding a new MADV_NO_1GHUGEPAGE hint so latency-sensitive applications can opt out on a per-VMA basis."

RVR_S = 'Opened a parallel sub-thread focused on NUMA topology interaction. On NUMA machines with 1TB+ per node, 1GB pages spanning NUMA boundaries would be catastrophically slow. Proposed that 1GB THP allocation must be NUMA-local-only with strict fallback to 2MB when cross-node placement would occur.'

DH2_S = "Strongly agreed with Rik's NUMA constraint. Further noted that this effectively means 1GB THP would be unusable on any machine where the NUMA topology is non-uniform at the 1GB granularity, which covers most real NUMA servers. Suggested this should be a hard kernel invariant enforced at allocation time, not a documentation caveat."

MW_S = 'Questioned the CMA reservation approach suggested by Zi Yan, arguing that reserving large contiguous regions at boot permanently removes that memory from the buddy allocator, creating a fragmentation cliff. Proposed instead using the large folio infrastructure that already exists to opportunistically promote to 1GB only when a naturally aligned region is available.'

DH  = {'author': 'David Hildenbrand', 'reply_to': '', 'summary': DH_S, 'sentiment': 'needs_work', 'sentiment_signals': ['requested changes', 'compaction concern'], 'has_inline_review': False, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': DH_B}
JW  = {'author': 'Johannes Weiner', 'reply_to': 'David Hildenbrand', 'summary': JW_S, 'sentiment': 'neutral', 'sentiment_signals': ['suggested alternative approach'], 'has_inline_review': True, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': JW_B}
ZY1 = {'author': 'Zi Yan', 'reply_to': 'Johannes Weiner', 'summary': ZY1_S, 'sentiment': 'positive', 'sentiment_signals': ['constructive suggestion', 'supported approach'], 'has_inline_review': True, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': ZY1_B}
ZY2 = {'author': 'Zi Yan', 'reply_to': 'David Hildenbrand', 'summary': ZY2_S, 'sentiment': 'neutral', 'sentiment_signals': ['addressed concern', 'proposed madvise extension'], 'has_inline_review': False, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': ZY2_B}
RVR = {'author': 'Rik van Riel', 'reply_to': '', 'summary': RVR_S, 'sentiment': 'needs_work', 'sentiment_signals': ['NUMA concern', 'strict requirement'], 'has_inline_review': False, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': RVR_B}
DH2 = {'author': 'David Hildenbrand', 'reply_to': 'Rik van Riel', 'summary': DH2_S, 'sentiment': 'needs_work', 'sentiment_signals': ['agreed with concern', 'stronger constraint proposed'], 'has_inline_review': False, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': DH2_B}
MW  = {'author': 'Matthew Wilcox', 'reply_to': 'Zi Yan', 'summary': MW_S, 'sentiment': 'needs_work', 'sentiment_signals': ['alternative approach preferred', 'CMA concern'], 'has_inline_review': False, 'tags_given': [], 'analysis_source': 'llm', 'raw_body': MW_B}

data = {
    'thread_id': '540c5c13-9cfb-44ea-b18f-8e4abff30a01@linux.dev',
    'subject': '[LSF/MM/BPF TOPIC] Beyond 2MB: Why Terabyte-Scale Machines Need 1GB Transparent Huge Pages',
    'url': 'https://lore.kernel.org/linux-mm/540c5c13-9cfb-44ea-b18f-8e4abff30a01@linux.dev/',
    'dates': {
        '2026-02-13': {
            'report_file': '2026-02-13_ollama_llama3.1-8b.html',
            'developer': 'Usama Arif',
            'analysis_source': 'llm-per-reviewer',
            'patch_summary': PATCH,
            'reviews': [DH, JW],
        },
        '2026-02-14': {
            'report_file': '2026-02-14_ollama_llama3.1-8b.html',
            'developer': 'Usama Arif',
            'analysis_source': 'llm-per-reviewer',
            'patch_summary': PATCH,
            'reviews': [DH, JW, ZY1, ZY2, RVR],
        },
        '2026-02-15': {
            'report_file': '2026-02-15_ollama_llama3.1-8b.html',
            'developer': 'Usama Arif',
            'analysis_source': 'llm-per-reviewer',
            'patch_summary': PATCH,
            'reviews': [DH, JW, ZY1, ZY2, RVR, DH2, MW],
        },
    },
}

pathlib.Path('C:/Users/krush/source/repos/lkml-reviews/sample_data.json').write_text(
    json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8'
)
print('JSON written, review counts:', {d: len(v['reviews']) for d, v in data['dates'].items()})
