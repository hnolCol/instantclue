import re

def buildRegex(categoriesList, withSeparator = True, splitString = None, matchingGroupOnly = False):
        '''
        Build regular expression that will search for the selected category. Importantly it will prevent 
        cross findings with equal substring
        =====
        Input:
            List of strings (categories) - userinput
        Returns:
            Regular expression that can be used. 
        ====

        '''
        regExp = r'' #init reg ex
        for category in categoriesList:
            category = re.escape(category) #escapes all special characters
            if withSeparator and splitString is not None:
                if matchingGroupOnly:
                    regExp = regExp + r'(?:{}{})|(?:^{}$)|(?:{}{}$)|'.format(category,splitString,category,splitString,category)
                else:
                    regExp = regExp + r'({}{})|(^{}$)|({}{}$)|'.format(category,splitString,category,splitString,category)
            else:
                if matchingGroupOnly:
                    regExp = regExp + r'(?:{})|'.format(category)
                else:
                    regExp = regExp + r'({})|'.format(category)
            
            
                

        regExp = regExp[:-1] #strip of last |

        return regExp
		
def buildReplaceDict(uniqueValues,splitSearchString):
    '''
    Subsets the currently selected df from dfClass on selected categories. Categories are 
    retrieved from user's selection.
    =====
    Input:
        uniuqeValues - unique values of a findall procedure. values of this
            object present the keys in the returned dict
        splitSearchString - List of strings that were entered by the user without quotes
    Returns:
        replace dict. can be used in map()
    ====
    '''
    replaceDict = dict() 
    naString = ''
    
    for value in uniqueValues:
        if all(x in value for x in splitSearchString):
            replaceDict[value] = splitSearchString
        if any(x in value for x in splitSearchString):
            repString = ''
            for category in splitSearchString:
                if category in value:
                    repString =repString + '{},'.format(category)
            replaceDict[value] = repString[:-1]
        else:
            replaceDict[value] = naString
    return replaceDict	