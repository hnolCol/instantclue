
from typing import Dict, AnyStr, Any
from ..utils.stringOperations import getMessageProps
import requests

GITHUB_URL = "https://api.github.com/repos/hnolcol/instantclue/releases"

class UpdateChecker:

    def __init__(self,version : str) -> None:
        self.version = version

    def checkForUpdates(self) -> Dict[AnyStr,Any]:
        ""
        try:
            response = requests.get(GITHUB_URL, timeout=2)
        except (requests.ConnectionError, requests.Timeout) as exception:
            return getMessageProps("Error ..","Timeout connection error when trying to check for a new version of InstantClue.")
        except Exception:
            return getMessageProps("Error ..","Unknown error when trying to check for updates.")
        if response.status_code == 200:
            try:
                data = response.json()
                if "tag_name" in data[0] and "html_url" in data[0]:
                    tagName = data[0]["tag_name"]
                    releaseURL = data[0]["html_url"]
                    if tagName != self.version:
                        funcKwargs = {"releaseURL" :  releaseURL }
                        return funcKwargs
                    else:
                        return getMessageProps("Done..","Instant Clue is up to date.")
                        
            except:
                getMessageProps("Error ..","Error when extracting the lates version from GitHub. Please check manually for updates.")
        else:
            return getMessageProps("Error ..",f"Error when connection to GitHub. Status code: {response.status_code}")
           