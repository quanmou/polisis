import os
import pandas as pd

cur_path = os.path.dirname(os.path.abspath(__file__))
data_Path = os.path.join(os.path.dirname(cur_path), 'data')
sanitized_polices_path = os.path.join(data_Path, 'OPP-115/sanitized_policies')


def get_pratices(file_path=sanitized_polices_path):
    pratices = {}
    for path, subdirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.html'):
                try:
                    with open(os.path.join(path, filename), 'r') as f:
                        policy_id, website = os.path.splitext(filename)[0].split('_')
                        polices = []
                        for line in f.readlines():
                            polices.extend(line.strip().split("|||"))
                        pratices[policy_id] = {
                            'website': website,
                            'polices': polices
                        }
                except Exception as e:
                    print(e)
    return pratices

