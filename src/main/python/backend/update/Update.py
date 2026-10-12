from typing import Dict, AnyStr, Any, Optional, Tuple
import re
from ..utils.stringOperations import getMessageProps
import requests

GITHUB_URL = "https://api.github.com/repos/hnolcol/instantclue/releases"

# Item 62: update check re-enabled with a real version comparison.
#
# The original check reported an update whenever the newest GitHub tag was
# merely DIFFERENT from the running version ("if tagName != self.version"), so
# any build whose version is not literally identical to the newest upstream tag
# (e.g. vF0.12.1.2, or v0.13.0 while GitHub's newest is v0.12.2) always claimed
# an update was available. It also took the first entry of the release list
# regardless of drafts/pre-releases, and its JSON-parsing error branch built a
# message without returning it (report item 1.5).
#
# Now: tags and the running version are parsed into numeric tuples
# ("v0.13.0" -> (0,13,0); a leading "v"/"vF" etc. is ignored), drafts and
# pre-releases are skipped, and an update is reported ONLY if the highest
# published release is strictly newer than the running version.


def parseVersion(versionString : str) -> Optional[Tuple[int,...]]:
    "Return e.g. (0,13,0) for 'v0.13.0', or None if no version number is found."
    if not isinstance(versionString,str):
        return None
    match = re.search(r"\d+(?:\.\d+)*", versionString)
    if match is None:
        return None
    parts = [int(x) for x in match.group(0).split(".")]
    #drop trailing zeros so that 0.13 == 0.13.0
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def isNewerVersion(candidate : str, current : str) -> bool:
    "True only if candidate is a strictly higher version number than current."
    c, r = parseVersion(candidate), parseVersion(current)
    if c is None or r is None:
        return False
    return c > r


class UpdateChecker:

    def __init__(self,version : str) -> None:
        self.version = version

    def getLatestRelease(self, releases) -> Optional[Dict[AnyStr,Any]]:
        "Highest-versioned published (non-draft, non-pre-release) release, or None."
        best, bestVersion = None, None
        for release in releases:
            if not isinstance(release,dict):
                continue
            if release.get("draft") or release.get("prerelease"):
                continue
            if "tag_name" not in release or "html_url" not in release:
                continue
            v = parseVersion(release["tag_name"])
            if v is None:
                continue
            if bestVersion is None or v > bestVersion:
                best, bestVersion = release, v
        return best

    def checkForUpdates(self, silent : bool = True) -> Dict[AnyStr,Any]:
        """Check GitHub for a newer release.

        Returns {"releaseURL", "latestVersion"} if a newer release exists.
        With silent=True (used at startup) nothing else is reported: no
        "up to date" note and no message when offline or GitHub is
        unreachable, so a normal start is never interrupted.
        """
        try:
            try:
                response = requests.get(GITHUB_URL, timeout=5)
            except (requests.ConnectionError, requests.Timeout):
                if silent: return {}
                return getMessageProps("Error ..","Timeout connection error when trying to check for a new version of InstantClue.")
            if response.status_code != 200:
                if silent: return {}
                return getMessageProps("Error ..",f"Error when connection to GitHub. Status code: {response.status_code}")
            latest = self.getLatestRelease(response.json())
            if latest is not None and isNewerVersion(latest["tag_name"], self.version):
                return {"releaseURL" : latest["html_url"], "latestVersion" : latest["tag_name"]}
            if silent: return {}
            return getMessageProps("Done..","Instant Clue is up to date.")
        except Exception:
            if silent: return {}
            return getMessageProps("Error ..","Error when extracting the latest version from GitHub. Please check manually for updates.")
