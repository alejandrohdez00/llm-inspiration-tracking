import json
import re
from xml.etree import ElementTree as etree
import os
import sys
from bs4 import BeautifulSoup

startdir = "" # The directory to find all the wiki files
outfilename = "" # The filename to dump plots to
titlefilename = "" # The filename to dump title names to

# Check command line parameters
if len(sys.argv) > 3:
	startdir = sys.argv[1]
	outfilename = sys.argv[2]
	titlefilename = sys.argv[3]
else:
	print("usage:", sys.argv[0], "directory plotfile titlefile")
	exit()

files = [] # All the wiki files

# Get all the wiki files left by wikiextractor
for dirname, dirnames, filenames in os.walk(os.path.join('.', startdir)):
	for filename in filenames:
		if filename[0] != '.':
			files.append(os.path.join(dirname, filename))

def is_movie_article(soup):
	"""Check if the article is about a movie"""
	infobox = soup.find('table', {'class': 'infobox'})
	if infobox:
		# Look for terms that indicate this is a movie
		text = infobox.get_text().lower()
		return any(term in text for term in ['directed by', 'film', 'movie', 'runtime', 'release date'])
	return False

def extract_movie_info(soup):
	"""Extract genre and director from movie infobox and introduction"""
	genre = []
	director = []
	
	# First try the infobox
	infobox = soup.find('table', {'class': 'infobox'})
	if infobox:
		# Find genre
		genre_row = infobox.find('th', string=re.compile(r'Genre|Genres'))
		if genre_row and genre_row.find_next_sibling('td'):
			genre = [g.strip() for g in genre_row.find_next_sibling('td').get_text().split(',')]
		
		# Find director
		director_row = infobox.find('th', string=re.compile(r'Directed by'))
		if director_row and director_row.find_next_sibling('td'):
			director = [d.strip() for d in director_row.find_next_sibling('td').get_text().split(',')]
	
	# If genre is empty, try to find it in the first paragraph
	if not genre:
		# Find the first paragraph
		first_p = soup.find('p')
		if first_p:
			text = first_p.get_text()
			
			# Common film genres
			genre_keywords = [
				'horror', 'comedy', 'drama', 'action', 'thriller', 'romance', 
				'science fiction', 'sci-fi', 'adventure', 'fantasy', 'western',
				'musical', 'documentary', 'animation', 'animated', 'supernatural',
				'crime', 'mystery', 'biographical', 'war', 'historical'
			]
			
			# Look for genre keywords in the text
			found_genres = []
			for keyword in genre_keywords:
				if keyword in text.lower():
					# Check if it's part of a compound genre (e.g., "supernatural horror")
					idx = text.lower().find(keyword)
					start = max(0, idx - 20)
					end = min(len(text), idx + len(keyword) + 20)
					context = text[start:end].lower()
					
					# Look for compound genres
					for other_keyword in genre_keywords:
						if other_keyword != keyword and other_keyword in context:
							if abs(context.find(other_keyword) - context.find(keyword)) < 15:
								compound = ' '.join(sorted([other_keyword, keyword]))
								if compound not in found_genres:
									found_genres.append(compound)
									break
					else:
						if keyword not in found_genres:
							found_genres.append(keyword)
			
			if found_genres:
				genre = found_genres
	
	# If director is empty, try to find it in the first paragraph
	if not director:
		first_p = soup.find('p')
		if first_p:
			text = first_p.get_text()
			directed_match = re.search(r'directed by ([^,.]+)', text, re.IGNORECASE)
			if directed_match:
				director = [directed_match.group(1).strip()]
	
	return {
		'genre': [g.title() for g in genre],  # Capitalize genre names
		'director': director
	}

with open(outfilename, "w") as outfile:
	# Opened the output file
	with open(titlefilename, "w") as titlefile:
		# Opened the title file
		# Walk through each file. Each file has a json for each wikipedia article. Look for jsons with "plot" subheaders
		for file in files:
			#print >> outfile, "file:", file #FOR DEBUGGING
			data = [] # Each element is a json record
			# Read the file and get all the json records
			for line in open(file, 'r'):
				data.append(json.loads(line))
			# Look for "plot" in a "h2" tag inside the text of the json
			for j in data:
				# j is a json record
				# Text element contains HTML
				soup = BeautifulSoup(j['text'].encode('utf-8'), "html.parser")
				
				# Only process if it's a movie article
				if not is_movie_article(soup):
					continue
				
				# Get movie information
				movie_info = extract_movie_info(soup)
				plot = "" # The plot found (if any)
				# Look for a "h2" tag that contains the word "plot"
				# The next element(s) will be the text of the plot
				inplot = False # Am I inside a plot section of the article?
				previousHeader = "" # What was the last subheading I saw? (plots may have sub-sub-headings)
				# Walk through each element in the html soup object
				for n in range(len(soup.contents)):
					current = soup.contents[n] # The current html element
					if current is not None and current.name == 'h2' and 'plot' in current.get_text().lower():
						# I found a plot header
						inplot = True
						previousHeader = previousHeader + current.get_text() + '. '
					elif inplot and current is not None and (current.name == 'h3' or current.name == 'h4'):
						# I'm probably seeing a sub-heading inside a plot block
						previousHeader = previousHeader + current.get_text() + '. '
					elif inplot and current is not None and (current.name is None or current.name == 'b' or current.name == 'a' or current.name == 'i' or current.name == 'strong' or current.name == 'em'):
						# I'm probably looking at text inside a plot block
						current = current.strip()
						# Sometimes we see the header name duplicated inside the text block that succeeds the sub-section header. Crop it off
						if len(current) > 0:
							if len(previousHeader) > 0:
								headerLength = len(previousHeader)
								plot = plot + current[headerLength:].strip() + ' '
							else:
								plot = plot + current.strip() + ' '
							# Forget the previous header. It was either consumed or wasn't duplicated in the first place.
							previousHeader = ""
					elif inplot and current is not None and (current.name == 'h1' or current.name == 'h2'):
						# Probably left the plot block. All done with this json!
						break
				# Did we find a plot?
				if len(plot) > 0:
					# Write movie info along with title
					movie_data = {
						'title': j['title'],
						'genre': movie_info['genre'],
						'director': movie_info['director'],
					}
					print(json.dumps(movie_data), file=titlefile)
					
					# Clean and write the plot
					# remove newlines
					plot = plot.replace('\n', ' ').replace('\r', '').strip()
					# remove html tags (probably mainly hyperlinks)
					plot = re.sub('<[^<]+?>', '', plot)
					# remove character name initials and take periods off mr/mrs/ms/dr/etc.
					plot = re.sub(' [M|m]r\.', ' mr', plot)
					plot = re.sub(' [M|m]rs\.', ' mrs', plot)
					plot = re.sub(' [M|m]s\.', ' ms', plot)
					plot = re.sub(' [D|d]r\.', ' dr', plot)
					#plot = re.sub(' [M|m]d\.', ' md', plot)
					#plot = re.sub(' [P|p][H|h][D|d]\.', ' phd', plot)
					#plot = re.sub(' [E|e][S|s][Q|q]\.', ' esq', plot)
					plot = re.sub(' [L|l][T|t]\.', ' lt', plot)
					plot = re.sub(' [G|g][O|o][V|v]\.', ' lt', plot)
					plot = re.sub(' [C|c][P|p][T|t]\.', ' cpt', plot)
					plot = re.sub(' [S|s][T|t]\.', ' st', plot)
					plot = re.sub(' [A-Z|a-z]\.', '', plot) # remove single letter initials
					plot = re.sub('\.\"', '\".', plot) # deal with periods in quotes
					# Acroymns with periods are not fun. Need two steps to get rid of those periods.
					# I don't think this is working quite right
					p1 = re.compile('([A-Z|a-z])\.([)|\"|\,])')
					plot = p1.sub(r'\1\2', plot)
					p2 = re.compile('\.([A-Z|a-z])')
					plot = p2.sub(r'\1', plot)
					# periods in numbers
					p3 = re.compile('([0-9]+)\.([0-9]+)')
					plot = p3.sub(r'\1\2', plot)
					# Break into sentences
					sentences = re.split('[\?\.\!]', plot)
					#print >> outfile, j['title'].encode('utf-8') #FOR DEBUGGING
					# Write the sentences to the plot file
					for s in sentences:
						if len(s.strip()) > 0:
							print(s.strip() + '.', file=outfile)
					print("<EOS>", file=outfile)