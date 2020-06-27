import os
import pandas as pd
from sklearn.utils import shuffle

cur_path = os.path.dirname(os.path.abspath(__file__))
data_Path = os.path.join(os.path.dirname(cur_path), 'data')
opp_path = os.path.join(data_Path, 'OPP-115')
sanitized_polices_path = os.path.join(opp_path, 'sanitized_policies')
consolidation_path = os.path.join(opp_path, 'consolidation')
threshold_05 = os.path.join(consolidation_path, 'threshold-0.5-overlap-similarity')
threshold_075 = os.path.join(consolidation_path, 'threshold-0.75-overlap-similarity')
threshold_1 = os.path.join(consolidation_path, 'threshold-1-overlap-similarity')

id2pratice = {0: "First Party Collection/Use", 1: "Third Party Sharing/Collection", 2: "User Choice/Control",
              3: "User Access, Edit and Deletion", 4: "Data Retention", 5: "Data Security", 6: "Policy Change",
              7: "Do Not Track", 8: "International and Specific Audiences", 9: "Other"}
practice2id = {"First Party Collection/Use": 0, "Third Party Sharing/Collection": 1, "User Choice/Control": 2,
               "User Access, Edit and Deletion": 3, "Data Retention": 4, "Data Security": 5, "Policy Change": 6,
               "Do Not Track": 7, "International and Specific Audiences": 8, "Other": 9}


def get_segments(file_path=sanitized_polices_path):
    segments = {}
    for path, subdirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.html'):
                try:
                    with open(os.path.join(path, filename), 'r') as f:
                        policy_id, website = os.path.splitext(filename)[0].split('_')
                        polices = []
                        for line in f.readlines():
                            polices.extend(line.strip().split("|||"))
                        segments[policy_id] = {
                            'website': website,
                            'polices': polices
                        }
                except Exception as e:
                    print(e)
    return segments


def process_segment_train_data(file_path=threshold_05):
    all_segments = get_segments()
    all_annotation_df = pd.DataFrame()
    for path, subdirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.csv'):
                policy_id, website = os.path.splitext(filename)[0].split('_')
                policy_segment = all_segments[policy_id]
                annotation_df = pd.read_csv(os.path.join(file_path, filename),
                                            names=['annotation_ID', 'batch_ID', 'annotator_ID', 'policy_ID',
                                                   'segment_ID', 'category_name', 'attribute-value', 'date',
                                                   'policy_URL'])
                annotation_df.insert(5, 'segment_content',
                                     [policy_segment['polices'][i] for i in annotation_df['segment_ID']])
                annotation_df.insert(6, 'category_id', [practice2id[i] for i in annotation_df['category_name']])
                all_annotation_df = pd.concat([all_annotation_df, annotation_df], axis=0)
    shuffled_annotation_df = shuffle(all_annotation_df)
    train_annotation_df = shuffled_annotation_df.iloc[:int(len(shuffled_annotation_df)*0.9), :]
    test_annotation_df = shuffled_annotation_df.iloc[int(len(shuffled_annotation_df)*0.9):, :]
    all_annotation_df.to_csv(os.path.join(data_Path, 'all_annotations.csv'), index=False)
    train_annotation_df.to_csv(os.path.join(data_Path, 'train_annotations.csv'), index=False)
    test_annotation_df.to_csv(os.path.join(data_Path, 'test_annotations.csv'), index=False)


# process_segment_train_data()
