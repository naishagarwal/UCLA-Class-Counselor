import pandas as pd
import json

# read in scraped data
df = pd.read_csv('~/cs263_final_project/263-project/followup_project/data/fall_2024/ucla_class_info_F_24.csv')

# filter df to get rid of unclean data
filtered = df[~df["class_times"].str.contains("am|pm") == False]
filtered = filtered[filtered['class_dates'].str.len() <= 5]

# check to see if class dates and times make sense
class_dates = filtered.class_dates.unique().tolist()
class_times = filtered.class_times.unique().tolist()

# write results to csv
filtered.to_csv('cleaned_ucla_class_info_F_24.csv', index=False)

# create dict for each record
d = filtered.to_dict('records')

# write results to file
with open('cleaned_ucla_class_info_F_24.json', 'w') as fp:
    json.dump(d, fp)