
# CHECK THE ACCIDENT_DATE.IPYNB to SEE WHAT ARE THE GLOBAL VARIALBLES

## STEP 1
# all index of the year of the accident 
all_dates = []
for i in range(len(tr_texts)):
    if y_tr_df['date_consolidation'][i] != 'n.c.':
        ff = [m.start() for m in re.finditer(y_tr_df['date_consolidation'][i][:4], tr_texts[i])]
    else: ff = []
    all_dates.append(ff)

## STEP 2
# check if there is the day digit near the year digit
sentences = []
for ind in range(len(tr_texts)):
    sentence = []
    window = 15
    for date in all_dates[ind]:
        left = tr_texts[ind][date - window : date]
        right = tr_texts[ind][date + 4 : date + 4 + window]
        left_num = re.findall(r'\d', left)
        right_num = re.findall(r'\d', right)
        if len(left_num) == 0 & len(right_num) == 0:
            pass
        else:
            text = tr_texts[ind][date - window : date + 4 + window]
            sentence.append(text)
    sentences.append(sentence)

## STEP 3 
# define custom pattern
pattern1 = r"\d{1,2} \w{3,9} \d{2,4}"
pattern2 = r"1° \w{3,9} \d{2,4}"
pattern3 = r"1er \w{3,9} \d{2,4}"
pattern4 = r"\d{1,2}/\d{1,2}/\d{2,4}"
pattern5 = r"\d{1,2} / \d{1,2} / \d{2,4}"
pattern6 = r"\d{1,2}/ \d{1,2}/ \d{2,4}"


pattern1_bis = r"(\d{1,2}) (\w{3,9}) (\d{2,4})"
pattern2_bis = r"(1°) (\w{3,9}) (\d{2,4})"
pattern3_bis = r"(1er) (\w{3,9}) (\d{2,4})"
pattern4_bis = r"(\d{1,2})/(\d{1,2})/(\d{2,4})"
pattern5_bis = r"(\d{1,2}) / (\d{1,2}) / (\d{2,4})"
pattern6_bis = r"(\d{1,2})/ (\d{1,2})/ (\d{2,4})"

patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6] 
patterns_bis = [pattern1_bis, pattern2_bis, pattern3_bis, pattern4_bis, pattern5_bis, pattern6_bis]


## STEP 4
# check the number of sentences that doesnt get detected using preious patterns
dates = []
fails = []
for i in range(len(sentences)):
    date = []
    index = []
    for sentence in sentences[i]:
        l = []
        for pattern in patterns:
            l.extend(re.findall(pattern, sentence))
        if len(l) == 0:
            fails.append(sentence)
        date.extend(l)
    dates.append(date)

## STEP 5
# Manually check sentences in the fails array and go back to STEP 3 
# and define a custom regex pattern that could detect the date in those sentences.
# Do it until no sentence with dates is left in the fails array. 