
# Class lookup table generation

# Load libraries

from datetime import date, timedelta

# OUR FORMAT
# D-DD/MM/YY

# date(YY, MM, DD)
# split(str=',')

# bday = date(1996,02,16)
# bday += timedelta(days=1)
# print(bday)

# date.strftime("%d/%m/%y")
# on linux or mac os, use -
# on windows, use #

# INPUT: string
# OUTPUT: list of dates

def lookupGen( str ):
	result = str.split(',')
	listOfRangeStrings = []
	for idx, myDate in enumerate(result):
		# nonzero means True
		if myDate.find("to") != -1: # handle range of dates
			# get two dates
			myDate = myDate.replace("to", "").strip()
			range = myDate.split()
			# create two date objects
			date1 = range[0].replace("D-", "").split("/") # DD MM YY
			date1 = date(1900 + int(date1[2]), int(date1[1]), int(date1[0]))
			date2 = range[1].replace("D-", "").split("/")
			date2 = date(1900 + int(date2[2]), int(date2[1]), int(date2[0]))
			# iterate through the range and add to classTwo
			it = date1
			while (it <= date2):
				result.append(it.strftime("D-%-d/%-m/%y"))
				it += timedelta(days=1)
			listOfRangeStrings.append(idx)
		else:
			result[idx] = myDate.strip()
	# cleaning up
	listOfRangeStrings.reverse()
	# print("Length:", len(listOfRangeStrings))
	for x in listOfRangeStrings:
		del result[x]
	return result





