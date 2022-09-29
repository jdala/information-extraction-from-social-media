import csv
import pandas as pd
from collections import defaultdict


parties = ['ΝΔ', 'ΣΥΡΙΖΑ', 'ΠΑΣΟΚ - ΚΙΝΑΛ', 'KKE', 'Ελληνική Λύση', 'ΜΕΡΑ25', 'Ανεξάρτητοι']

# database_ids = pd.ExcelFile('database_nodes-old.xlsx').parse()[['id', 'twitter_id', 'handles']]
database_ids = pd.read_csv('database_nodes.csv')[['id', 'twitter_id', 'handles']]

id_handle_mapping = defaultdict(lambda: [None, None])
for ind, val in enumerate(database_ids['handles']):
    for handle in val.split(','):
        id_handle_mapping[handle.strip()] = [database_ids['id'][ind], database_ids['twitter_id'][ind]]


total_found = total_not_found = 0
with open('./mapping-not_NaN.csv', 'w') as f:
    writter = csv.DictWriter(f, delimiter=',', fieldnames=['twitter_handle', 'twitter_id', 'id', 'party'])
    writter.writeheader()

    for party_index, party in enumerate(parties):
        party_members = pd.ExcelFile('noi_list.xlsx').parse(party)[['Twitter', 'Ελληνική Βουλή']].to_numpy()
        greek_parliament = list(filter(lambda member: member[1] == 1, party_members))

        found = not_found = 0

        for member in greek_parliament:
            if isinstance(member[0], str):
                twitter_handle = member[0].split('/')[-1].strip().split('?')[0]
                [db_id, twitter_id] = id_handle_mapping[twitter_handle]
                # writter.writerow({'twitter_handle': twitter_handle, 'twitter_id': twitter_id, 'id': db_id, 'party': party_index})
                
                if twitter_id == None: 
                    total_not_found += 1
                    not_found += 1
                else:
                    writter.writerow({'twitter_handle': twitter_handle, 'twitter_id': twitter_id, 'id': db_id, 'party': party_index})

                    total_found += 1
                    found += 1

        print(f'Παράταξη: {party} ({party_index})')
        print(f'Found: {found} \nNot found: {not_found}\n')

print(f'====================\nΣυνολικά:')
print(f'Found: {total_found} \nNot found: {total_not_found}\n')