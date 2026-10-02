# CosyVoice2 bring-up — STATUS (2026-10-02: merged, bounty complete)

Rewrite this file each session; don't append to it. Every claim here cites a commit, a command, or an ID in
FINDINGS.md or DECISIONS.md (D*). The previous STATUS (09-30, with 10-01's in-progress block) is
`history/STATUS_2026-09-30_and_10-01.md`.

## Done

- **#56651 merged** into tenstorrent/tt-metal main on 2026-10-01 at 15:25 UTC, by mtairum, as `98a19fd851`. The
  merged head is `33aa3601eb`.
  - Approved by mtairum (09-30, after Tier-3 CI on `wh_n150`, run 36746885220) and by tdowdallTT (10-01). Then the
    merge queue took it.
  - The merge commit's message is the PR description of 09-15 ("This is a draft ..."). The queue took the message
    at enqueue time (14:19), and the user's new description went up at 15:09. The PR page shows the new one
    (`drafts/2026-10-01_pr_56651_description_for_merge.md`). The commit also carries one `Co-authored-by: Claude
    Sonnet 5` line, from the 09-09/10 commits.
- **#54104 closed, bounty complete** (jberkowitzTT, 10-01 15:56). Next step: the user emails
  bounties@tenstorrent.com with a link to the issue, to start the payment.

## Not merged (optional; the user's call)

- `backup/2026-09-29-pr` holds three commits after the merged head: `ea95ce1d66` (stage A's last-0.1 s check,
  B38), `1dcdc10059` (the Euler step count as a config option, B39), and `213afe7909` (the heads merged by
  `nlp_concat_heads`, B40).
  - B40 is bit-identical output and 4–6 % lower RTF.
  - A follow-up PR would need a rebase onto main and a re-run on a card without B44's problem.
- Lever (b), the traced CFM during streaming, is a proposal with a prototype (B41,
  `design/2026-09-30_traced_cfm_streaming.md`).
- B45, stage A's last-0.1 s check at digital silence, is open.

## The 10-01 pod

- `app-7a544cdc-deployment-65d799ff57-9l4dj`, n150 L at `0000:e1:00.0` (B42).
- The harness killed a device job at its 30-minute limit, and chains now start with `scripts/2026-10-01/detach.sh`
  (B43).
- This card's LLM decode is not bitwise reproducible (B44). Use another card for any further measurement.
