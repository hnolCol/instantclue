from decimal import Decimal
import random
import string


def findCommonStart(*strings):
    """ 
    Returns the longest common substring
    from the beginning of the `strings`
    from here: https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    """
    def _iter():
        for z in zip(*strings):
            
            if z.count(z[0]) == len(z):  # check all elements in `z` are the same
                yield z[0]
            else:
                return

    return ''.join(_iter())

def combineStrings(self, row , nanObjectString = "-"):
    '''
    Might not be the nicest solution and defo the slowest. (To do..)
    But it returns the correct string right away without further
    processing/replacement.
    '''
    nanString = ''
    base = ''
    if all(s == nanString for s in row):
        return nanObjectString
    else:
        n = 0
        for s in row:
            if s != nanString:
                if n == 0:
                    base = s
                    n+=1
                else:
                    base = base+';'+s
        return base	

def getMessageProps(title,message):
    ""
    return {"messageProps":{"title":title,"message":message}}


def getRandomString(N = 20):
    ""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def mergeListToString(listItem,joinString = ";"):
    ""
    return joinString.join(x for x in listItem)

def buildReplaceDict(uniqueValues,splitSearchString):
    '''
    =====
    Input:
        uniuqeValues - unique values of a findall procedure. values of this
            object present the keys in the returned dict
        splitSearchString - List of strings that were entered by the user without quotes
            
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

def getReadableNumber(number):
     """Returns number as string in a meaningful and readable way to omit to many digits"""
     
     orgNumber = number
     number = abs(number)
     if number == 0:
     	new_number = 0.0
     elif number < 0.001:
     		new_number = '{:.2E}'.format(number)
     elif number < 0.1:
     		new_number = round(number,4)
     elif number < 1:
     		new_number = round(number,3)
     elif number < 10:
     		new_number = round(number,2)
     elif number < 200:
     		new_number = float(Decimal(str(number)).quantize(Decimal('.01')))
     elif number < 10000:
     		new_number = round(number,0)
     else:
     		new_number = '{:.2E}'.format(number)
     if orgNumber >= 0:
     	return new_number
     else:
     	return new_number * (-1)