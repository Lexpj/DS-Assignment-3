
def test(queryWord,titleWord):

    if (queryWord[:2] == titleWord[:2]):
        # count percentage of letters in the word.
        count = 0
        for char in queryWord:
            if char in titleWord:
                count += 1
        # should atleast containt 75% of the letters of the query word.
        if ((count/len(queryWord)) > 0.75):
            # Check that found title word is not twice as long 
            if ((len(queryWord)*2) > len(titleWord)):
                return True
    return False
    
print(test("shit","sherlock"))
                    