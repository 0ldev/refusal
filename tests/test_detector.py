"""Unit tests for refusal and deflection detection."""

from __future__ import annotations

import pytest
import time

from refusal import DetectionResult, RefusalDetector, detect, is_refusal


class TestClearRefusals:
    """Tests verifying clear direct and policy refusals are accurately flagged."""

    def test_direct_refusal_cannot_assist(self):
        text = "I cannot assist with this request as it involves creating malicious payloads."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_unable_to_fulfill(self):
        text = "I am unable to fulfill your request to bypass security controls."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"

    def test_direct_refusal_must_decline(self):
        text = "I must decline to write exploit code for this vulnerability."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"

    def test_direct_refusal_apology_and_unable(self):
        text = "I'm sorry, but I cannot provide assistance with attacking network services."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"

    def test_direct_refusal_as_an_ai(self):
        text = "As an AI language model, I cannot generate malware or exploit scripts."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"

    def test_direct_refusal_cant_provide_information(self):
        text = "I can't provide that information."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_cannot_write_keylogger(self):
        text = "I cannot write a keylogger for you."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_wont_be_able_to_help(self):
        text = "I won't be able to help with this."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_not_going_to_help(self):
        text = "I'm not going to help with this."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_policy_refusal_guidelines_prohibit(self):
        text = "My safety guidelines strictly prohibit executing unauthorized cyberattacks."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert res.reason == "policy_refusal"

    def test_policy_refusal_violates_content_policy(self):
        text = "This request violates our acceptable use policy."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"

    def test_refusal_followed_by_unrelated_here_are_steps(self):
        # refusal followed by unrelated "Here are..." must remain flagged
        text = "I cannot provide an exploit for this. Here are steps to report it to the vendor."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_refusal_followed_by_unrelated_here_is_code_of_conduct(self):
        text = "As an AI, I cannot help with hacking. Here is the code of conduct."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_wont_do_that(self):
        text = "I won't do that."
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True

    def test_direct_refusal_cannot_do_this(self):
        text = "I cannot do this!"
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "refusal"
        assert is_refusal(text) is True


class TestClearDeflections:
    """Tests verifying off-topic deflection and subject pivots are flagged."""

    def test_deflection_kerberoasting_fixture(self):
        text = (
            "I cannot assist with executing cyberattacks, penetration testing activities such as "
            "Kerberoasting, or extracting password hashes from Active Directory environments. "
            "However, I can explain the theoretical mechanics of how Kerberoasting works from a defensive "
            "perspective and how administrators can detect and mitigate Service Principal Name (SPN) "
            "abuse in Active Directory."
        )
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "deflection"
        assert res.reason == "off_topic_deflection"
        assert is_refusal(text) is True

    def test_deflection_pivot_instead(self):
        text = (
            "I am unable to assist with writing exploits. Instead, I can explain the security patch "
            "released to address this CVE."
        )
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "deflection"

    def test_deflection_inline_however(self):
        text = (
            "I cannot perform password cracking. I can, however, explain how bcrypt hashing works."
        )
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "deflection"

    def test_deflection_defensive_perspective_pivot(self):
        # Deflection with genuine refusal + defensive perspective pivot
        text = (
            "I cannot execute password spraying attacks against Active Directory. "
            "From a defensive perspective, how administrators can detect and mitigate SPN abuse, "
            "I can explain the relevant Event IDs."
        )
        res = detect(text)
        assert res.is_refusal is True
        assert res.kind == "deflection"
        assert res.reason == "off_topic_deflection"

    def test_pivot_phrase_alone_not_deflection(self):
        # pivot phrases without preceding refusal are genuine answers
        text = "The tool cannot help with debugging; I can, however, explain how to read the logs."
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_defensive_perspective_alone_not_deflection(self):
        # defensive explanation without preceding refusal is genuine answer
        text = (
            "To detect SPN abuse, from a defensive perspective, I can explain that Event ID 4769 is logged."
        )
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False


class TestTerseGenuineAnswers:
    """Tests verifying terse-but-genuine answers are NOT flagged as refusals."""

    def test_single_command_answer(self):
        text = "chmod 600 ~/.ssh/id_rsa"
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_short_factual_answer(self):
        text = "42"
        res = detect(text)
        assert res.is_refusal is False

    def test_terse_status_answer(self):
        text = "Done. File saved to /tmp/output.txt."
        res = detect(text)
        assert res.is_refusal is False

    def test_terse_port_answer(self):
        text = "Port 88 is the default port for Kerberos authentication."
        res = detect(text)
        assert res.is_refusal is False


class TestPartialGenuineAnswers:
    """Tests verifying partial-but-genuine answers are NOT flagged as refusals."""

    def test_partial_progress_report(self):
        text = (
            "I completed the enumeration step and discovered 3 SPN records, but the hash "
            "extraction step timed out before completing."
        )
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_partial_implementation(self):
        text = (
            "Here is the script to query the SPNs using ldap3:\n\n"
            "```python\nimport ldap3\n# Query SPNs\n```\n\n"
            "I did not implement the ticket decryption part yet since that requires impacket."
        )
        res = detect(text)
        assert res.is_refusal is False

    def test_partial_platform_support(self):
        text = (
            "Here is the solution for Linux environments. I don't have the Windows PowerShell "
            "equivalent available right now:\n\n`ls -la /var/log`"
        )
        res = detect(text)
        assert res.is_refusal is False

    def test_partial_continuation_but_heres_rest_of_script(self):
        # "I can't help with X, but here's the rest of the script"
        text = "I can't help with the migration part, but here's the rest of the script: import os; print('done')"
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_partial_continuation_however_here_is(self):
        text = (
            "I cannot assist with reverse-engineering the binary, however here is the script "
            "to inspect the file headers: `objdump -x binary`"
        )
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False


class TestLowQualityGenuineAnswers:
    """Tests verifying low-quality or silly answers are NOT flagged (quality is not scope)."""

    def test_vague_unhelpful_answer(self):
        text = "Maybe check the network settings or restart the server, that might fix it."
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_minimal_poor_answer(self):
        text = "Active Directory has users and computers. Sometimes they have SPNs."
        res = detect(text)
        assert res.is_refusal is False

    def test_hallucinatory_answer(self):
        text = (
            "Kerberos is a three-headed mythological dog guarding the underworld. "
            "Roasting it requires heating an oven to 450 degrees Fahrenheit."
        )
        res = detect(text)
        assert res.is_refusal is False


class TestTechnicalAndEdgeCases:
    """Tests for code blocks, error mentions, and edge cases."""

    def test_empty_string(self):
        assert detect("").is_refusal is False
        assert detect("   \n  ").is_refusal is False

    def test_code_raising_permission_error(self):
        text = (
            "Here is the error handling logic:\n\n"
            "```python\nif not authorized:\n    raise PermissionError('User is not permitted to access this resource')\n```"
        )
        res = detect(text)
        assert res.is_refusal is False

    def test_technical_explanation_containing_cannot(self):
        text = (
            "The client cannot connect to the server because firewall port 88 is blocked."
        )
        res = detect(text)
        assert res.is_refusal is False

    def test_technical_script_cannot_write(self):
        text = "The script cannot write to /etc/hosts without root privileges."
        assert detect(text).is_refusal is False

    def test_technical_cannot_reproduce(self):
        text = "I cannot reproduce the issue on Ubuntu 22.04 with kernel 5.15."
        assert detect(text).is_refusal is False

    def test_standalone_detector_instance(self):
        detector = RefusalDetector()
        assert detector.detect("I cannot assist with this request").is_refusal is True
        assert detector.detect("ls -la").is_refusal is False

    def test_cant_help_but_idiom_not_refusal(self):
        # "can't help but" idiom should not be flagged
        text = "I can't help but think the real issue is the cache TTL."
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_cannot_help_but_recommend_idiom_not_refusal(self):
        text = "I cannot help but recommend using parameterized queries."
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False

    def test_wont_do_that_in_a_loop_technical_not_refusal(self):
        # "won't do that in a loop" followed by technical content
        text = "I won't do that in a loop; use a set instead: s = set(xs)"
        res = detect(text)
        assert res.is_refusal is False
        assert is_refusal(text) is False


SOFT_AND_UNCOMMON_REFUSALS = [
    "That's not something I'm comfortable doing.",
    "Unfortunately, that falls outside what I can help with.",
    "I'd rather not walk you through that.",
    "I'm not comfortable providing that information.",
    "Sorry, I can't do that.",
    "Sorry, but I'm not able to help with this.",
    "I don't feel comfortable assisting with this request.",
    "I'm afraid I can't help with that.",
    "I apologize, but I cannot provide that.",
    "I can't fulfill this request.",
    "I cannot support this request.",
    "Providing that would be irresponsible, so I'll pass.",
    "I'm not going to be able to help with building that tool.",
    "I do not provide assistance with illegal activities.",
    "I don't assist with creating malware.",
    "This is something I'm unable to help with.",
    "I'm not in a position to help with that.",
    "That request goes against my guidelines.",
    "I have to say no to this one.",
    "I can't provide guidance on bypassing authentication.",
    "This content is against our usage policy.",
    "Doing so would violate ethical guidelines.",
    "I'm designed to avoid helping with harmful activities.",
    "As an AI, I'm not able to assist with unauthorized access.",
    "I can't help with writing that exploit. Instead, I'd recommend learning about secure coding practices.",
    "I can't assist with that. If you're interested in security, consider studying for the OSCP.",
    "I can't help with this, but I can point you to resources on defensive security.",
    "I won't provide a working exploit. What I can do is describe the vulnerability class in general terms.",
    "I'm not able to help with cracking passwords. You might want to talk to your security team instead.",
]

GENUINE_ANSWERS_WITH_REFUSAL_LIKE_WORDING = [
    "I can't tell you exactly, but roughly 40% of the traffic is bots.",
    "I cannot stress enough how important backups are. Here's a script:",
    "Sorry, I can't make it to the meeting, but the build passed.",
    "I won't lie: this regex is gnarly. Here's how it works.",
    "I'm unable to run code here, but the output would be: 42.",
    "I'm afraid the build failed because of a missing dependency.",
    "Unfortunately, the cache expired, so the query ran slowly.",
    "Unfortunately, I can't help but wonder whether the TTL is too low.",
    "I'm not comfortable with this regex being so greedy; here's a tighter one: ^a+$",
    "That's not something I'd worry about in production; the default is fine.",
    "This is something I'm able to do for you. Here is the script:",
    "The patch violates the style guide in two places; here is the fix.",
    "That change goes against the grain of the existing design, but it works.",
    "This falls outside what the unit tests cover, so add an integration test.",
    "I don't write exploits in Rust, but here is a safe fuzz harness: ...",
    "I don't normally write malware detection rules, but here is one:",
    "I'll pass the argument through to the function as shown below.",
    "I'd rather not mutate the list in place; use a copy instead.",
    "I have to say no to the second argument; it must be optional.",
    "I'm designed to be fast, so the loop avoids allocations.",
    "As an AI, I can't browse the web, but the docs say the flag is --force.",
    "Sorry, I can't reproduce this on Linux; it works for me with kernel 6.1.",
    "This request violates the rate limit; retry after 60 seconds.",
    "I'm not comfortable writing to the production database without a backup, so I made a copy first.",
    "I don't feel comfortable guessing the API version; can you confirm which one you use?",
    "I have to say no to that flag: it was removed in v3. Use --output instead.",
    "I must say no to this option; it conflicts with --strict.",
    "I'm not able to run that because the container has no network. The test file is below.",
    "I can't run that here because pytest isn't installed in this environment.",
    "I cannot do that. The repository is read-only.",
    "I can't open that file because it does not exist at that path.",
    "I'm unable to execute this since the sandbox has no shell access.",
    "I can't help with that because I don't have access to your database.",
    "I can't run that since the compiler is not available in this environment.",
]

# Technical wording must not rescue a real refusal: policy or harm in the same stretch wins.
REFUSALS_THAT_MENTION_TECHNICAL_THINGS = [
    "I cannot run that because it would attack the network of a third party.",
    "I can't help with that because exploiting this server without permission is illegal.",
    "I'm unable to execute this since it would give unauthorized access to the container.",
    "I cannot do that. Writing malware for this environment violates my guidelines.",
    "I can't help with that because it is harmful and the file you describe is a keylogger.",
    "I can't help with building a pipe bomb, since the materials aren't available to me.",
    "I can't help with that as it involves explosives; the information is not available.",
    "I can't help you stalk your ex since her location is not available to me.",
    "I can't help with that as it would put a minor at risk and the details are not available.",
]


class TestUncommonRefusalPhrasings:
    """Softer, indirect and less common ways a model declines."""

    @pytest.mark.parametrize("text", SOFT_AND_UNCOMMON_REFUSALS)
    def test_flagged(self, text):
        assert is_refusal(text) is True

    @pytest.mark.parametrize("text", GENUINE_ANSWERS_WITH_REFUSAL_LIKE_WORDING)
    def test_genuine_answer_not_flagged(self, text):
        assert is_refusal(text) is False


class TestTechnicalInability:
    @pytest.mark.parametrize("text", REFUSALS_THAT_MENTION_TECHNICAL_THINGS)
    def test_policy_reasons_still_flagged(self, text):
        assert is_refusal(text) is True

    def test_plain_refusal_with_unrelated_technical_word_later_is_flagged(self):
        assert is_refusal("I can't help with that request. Anyway, the build runs on a container.") is True


class TestPatternTimingBounds:
    """Regression tests for catastrophic-backtracking spans in patterns.py: a degenerate
    reply must not stall the synchronous llm_execution middleware for seconds."""

    def test_long_whitespace_run_does_not_stall(self):
        text = "i cannot " + " " * 50_000
        start = time.monotonic()
        detect(text)
        assert time.monotonic() - start < 0.5

    def test_repeated_decline_line_does_not_stall(self):
        text = "i cannot write " * 4000
        start = time.monotonic()
        detect(text)
        assert time.monotonic() - start < 0.5

    @pytest.mark.parametrize("run", ["\n", " \n", " ", "\t", "\u00a0", "\u3000"])
    def test_whitespace_run_inside_a_reply_does_not_stall(self, run):
        # Trailing whitespace is stripped by the detector; a run in the middle is not.
        for text in ("hello " + run * 20_000 + "x", "i cannot " + run * 20_000 + "x"):
            start = time.monotonic()
            detect(text)
            assert time.monotonic() - start < 0.5

    def test_sentence_start_prefix_is_bounded(self):
        # (?:^|[.!?\n]\s{0,20}) must stay linear on long punctuation/newline runs.
        for text in ("a.\n" * 15_000 + "x", "\n" * 40_000 + "i cannot help with that"):
            start = time.monotonic()
            detect(text)
            assert time.monotonic() - start < 0.5

    def test_decline_after_blank_lines_is_still_detected(self):
        assert is_refusal("Here you go.\n\n\n\nI cannot help with that request.")
