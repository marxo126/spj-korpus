"""WCAG 1.1.1, 1.2.1-5, 1.4.2 — Media accessibility rules."""
from __future__ import annotations

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, has_accessible_name


class MediaRule(BaseRule):
    id = "media"
    name = "Media"
    wcag_criteria = ("1.1.1", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.4.2")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_img_alt(ctx))
        findings.extend(self._check_decorative_alt(ctx))
        findings.extend(self._check_video_track(ctx))
        findings.extend(self._check_video_autoplay(ctx))
        findings.extend(self._check_svg_accessible(ctx))
        return findings

    def _check_img_alt(self, ctx) -> list[Finding]:
        """All <img> must have alt attribute."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "img":
                continue
            if "alt" not in elem.attributes:
                findings.append(self._finding(
                    check_id="img-alt",
                    severity=Severity.CRITICAL,
                    wcag="1.1.1",
                    wcag_name="Non-text Content",
                    message="<img> missing alt attribute",
                    file=path,
                    line=elem.line,
                    element=f"<img src=\"{elem.attributes.get('src', '...')}\">",
                    fix="Add alt attribute with descriptive text, or alt=\"\" for decorative images",
                    impact=("blind",),
                ))
        return findings

    def _check_decorative_alt(self, ctx) -> list[Finding]:
        """Images with alt="" should have role=presentation or aria-hidden."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "img":
                continue
            alt = elem.attributes.get("alt")
            if alt is None or str(alt).strip() != "":
                continue
            role = str(elem.attributes.get("role", "")).lower()
            aria_hidden = str(elem.attributes.get("aria-hidden", "")).lower()
            if role not in ("presentation", "none") and aria_hidden != "true":
                findings.append(self._finding(
                    check_id="decorative-alt",
                    severity=Severity.MINOR,
                    wcag="1.1.1",
                    wcag_name="Non-text Content",
                    message="<img alt=\"\"> without role=\"presentation\" or aria-hidden",
                    file=path,
                    line=elem.line,
                    element="<img alt=\"\">",
                    fix="Add role=\"presentation\" or aria-hidden=\"true\" for decorative images",
                ))
        return findings

    def _check_video_track(self, ctx) -> list[Finding]:
        """<video> must have <track kind=captions> or aria-label."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "video":
                continue
            # Check children for <track kind="captions">
            has_captions = False
            for child in elem.children:
                if child.tag.lower() == "track":
                    kind = str(child.attributes.get("kind", "")).lower()
                    if kind == "captions":
                        has_captions = True
                        break
            if has_captions:
                continue
            if has_accessible_name(elem):
                continue
            aria_described = elem.attributes.get("aria-describedby")
            if aria_described:
                continue
            findings.append(self._finding(
                check_id="video-track",
                severity=Severity.SERIOUS,
                wcag="1.2.2",
                wcag_name="Captions (Prerecorded)",
                message="<video> without captions track or accessible description",
                file=path,
                line=elem.line,
                element="<video>",
                fix="Add <track kind=\"captions\" src=\"...\" srclang=\"sk\">",
                impact=("deaf",),
            ))
        return findings

    def _check_video_autoplay(self, ctx) -> list[Finding]:
        """<video> with autoplay must have muted."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "video":
                continue
            has_autoplay = "autoplay" in elem.attributes
            has_muted = "muted" in elem.attributes
            if has_autoplay and not has_muted:
                findings.append(self._finding(
                    check_id="video-autoplay",
                    severity=Severity.MODERATE,
                    wcag="1.4.2",
                    wcag_name="Audio Control",
                    message="<video autoplay> without muted attribute",
                    file=path,
                    line=elem.line,
                    element="<video autoplay>",
                    fix="Add muted attribute to autoplaying videos",
                    impact=("cognitive", "vestibular"),
                ))
        return findings

    def _check_svg_accessible(self, ctx) -> list[Finding]:
        """<svg> must have <title>, aria-label, or aria-hidden."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "svg":
                continue
            # Check for aria-hidden (decorative)
            aria_hidden = str(elem.attributes.get("aria-hidden", "")).lower()
            if aria_hidden == "true":
                continue
            # Check for aria-label
            if has_accessible_name(elem):
                continue
            # Check children for <title>
            has_title = any(c.tag.lower() == "title" for c in elem.children)
            if has_title:
                continue
            role = str(elem.attributes.get("role", "")).lower()
            if role in ("presentation", "none"):
                continue
            findings.append(self._finding(
                check_id="svg-accessible",
                severity=Severity.MODERATE,
                wcag="1.1.1",
                wcag_name="Non-text Content",
                message="<svg> without accessible name or aria-hidden",
                file=path,
                line=elem.line,
                element="<svg>",
                fix="Add <title>, aria-label, or aria-hidden=\"true\"",
                impact=("blind",),
            ))
        return findings
