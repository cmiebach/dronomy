"""Pluggable image matchers. The brief asks us to compare >=2 approaches:
a classical one (SIFT/ORB) and a modern deep one (LoFTR/SuperGlue).

Every matcher returns a `MatchResult` (corresponding keypoints + a homography),
so the localization pipeline is agnostic to which matcher produced it.
"""
from .base import Matcher, MatchResult, get_matcher  # noqa: F401
