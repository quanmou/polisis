import os
import json
import pandas as pd
from sklearn.utils import shuffle

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(cur_path), 'data')
opp_path = os.path.join(data_path, 'OPP-115')
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
    all_segment_df = pd.DataFrame()
    all_annotation_df = pd.DataFrame()
    for path, subdirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.csv'):
                policy_id, website = os.path.splitext(filename)[0].split('_')
                policy_segment = all_segments[policy_id]
                annotation_df = pd.read_csv(os.path.join(file_path, filename),
                                            names=['annotation_ID', 'batch_ID', 'annotator_ID', 'policy_ID',
                                                   'segment_ID', 'category_name', 'attribute_value', 'date',
                                                   'policy_URL'])
                annotation_df.insert(5, 'segment_content',
                                     [policy_segment['polices'][i] for i in annotation_df['segment_ID']])
                annotation_df.insert(6, 'category_ID', [practice2id[i] for i in annotation_df['category_name']])
                segment_categories = {a: set(x) for a, x in annotation_df.groupby('segment_ID')['category_ID']}
                segment_categories = {segment: ','.join([str(int(i in category)) for i in range(len(id2pratice))])
                                      for segment, category in segment_categories.items()}
                segment_df = annotation_df[['policy_ID', 'segment_ID', 'segment_content']]
                segment_df = segment_df.drop_duplicates(['segment_ID'])
                segment_df.insert(0, 'filename', filename)
                segment_df['category_ID'] = [segment_categories[seg] for seg in segment_df['segment_ID']]

                all_annotation_df = pd.concat([all_annotation_df, annotation_df], axis=0)
                all_segment_df = pd.concat([all_segment_df, segment_df], axis=0)

    # shuffled_annotation_df = shuffle(all_annotation_df)
    # train_annotation_df = shuffled_annotation_df.iloc[:int(len(shuffled_annotation_df)*0.9), :]
    # test_annotation_df = shuffled_annotation_df.iloc[int(len(shuffled_annotation_df)*0.9):, :]
    # all_annotation_df.to_csv(os.path.join(data_path, 'all_annotations.csv'), index=False)
    # train_annotation_df.to_csv(os.path.join(data_path, 'train_annotations.csv'), index=False)
    # test_annotation_df.to_csv(os.path.join(data_path, 'test_annotations.csv'), index=False)

    shuffled_segment_df = shuffle(all_segment_df)
    train_segment_df = shuffled_segment_df.iloc[:int(len(shuffled_segment_df)*0.8), :]
    validation_segment_df = shuffled_segment_df.iloc[int(len(shuffled_segment_df)*0.8):int(len(shuffled_segment_df)*0.9), :]
    test_segment_df = shuffled_segment_df.iloc[int(len(shuffled_segment_df)*0.9):, :]
    all_segment_df.to_csv(os.path.join(data_path, 'all_segment.csv'), index=False)
    train_segment_df.to_csv(os.path.join(data_path, 'train_segment.csv'), index=False)
    validation_segment_df.to_csv(os.path.join(data_path, 'validation_segment.csv'), index=False)
    test_segment_df.to_csv(os.path.join(data_path, 'test_segment.csv'), index=False)


# process_segment_train_data()

def get_unique_value():
    all_annotation_df = pd.read_csv(os.path.join(data_path, 'all_annotations.csv'))
    category_df = all_annotation_df[['segment_content', 'category_name', 'attribute_value']]
    first_party_df = category_df.loc[category_df['category_name'] == "First Party Collection/Use"]
    third_party_df = category_df.loc[category_df['category_name'] == "Third Party Sharing/Collection"]
    user_choice_df = category_df.loc[category_df['category_name'] == "User Choice/Control"]
    user_access_df = category_df.loc[category_df['category_name'] == "User Access, Edit and Deletion"]
    data_retention_df = category_df.loc[category_df['category_name'] == "Data Retention"]
    data_security_df = category_df.loc[category_df['category_name'] == "Data Security"]
    policy_change_df = category_df.loc[category_df['category_name'] == "Policy Change"]
    do_not_track_df = category_df.loc[category_df['category_name'] == "Do Not Track"]
    audience_df = category_df.loc[category_df['category_name'] == "International and Specific Audiences"]
    other_df = category_df.loc[category_df['category_name'] == "Other"]

    # Extract Personal Information Type
    cols = ['segment_content', 'attributes']
    pers_info_type_df = pd.DataFrame(columns=cols)
    attributes = ['startIndexInSegment', 'endIndexInSegment', 'selectedText', 'value']
    for df in [first_party_df, third_party_df, user_choice_df]:
        seg_attributes = {}
        for row in df.iterrows():
            seg_content = row[1][0]
            all_attributes = json.loads(row[1][2])
            pers_attr = all_attributes.get('Personal Information Type')
            pers_val = [pers_attr.get(attr, '') for attr in attributes]
            if seg_content not in seg_attributes:
                seg_attributes[seg_content] = [pers_val]
            else:
                seg_attributes[seg_content].append(pers_val)

        for seg, attr in seg_attributes.items():
            attr.sort(key=lambda x: x[0])
            merged = []
            for ar in attr:
                if ar[0] == -1:  # filter startIndex and endIndex is -1
                    continue
                if not merged or merged[-1][1] <= ar[0]:
                    merged.append(ar)
                else:
                    if ar[1] <= merged[-1][1]:
                        continue
                    if merged[-1][3] == ar[3]:
                        merged[-1][2] = merged[-1][2] + ar[2][merged[-1][1] - ar[1]:]
                        merged[-1][1] = max(merged[-1][1], ar[1])
                    else:
                        ar[2] = ar[2][merged[-1][1] - ar[1]:]
                        ar[0] = merged[-1][1]
                        if ar[2] != ',' and ar[2] != ' ':
                            merged.append(ar)
            attr_str = '\n'.join(['☀'.join([str(m) for m in merg]) for merg in merged])
            if attr_str:
                pers_info_type_df.loc[pers_info_type_df.shape[0] + 1] = [seg] + [attr_str]

    pers_info_type_df.to_csv(os.path.join(data_path, 'personal_information_type_attributes.csv'), index=False)
    shuffle(pers_info_type_df)
    pers_info_type_train_df = pers_info_type_df.iloc[:int(len(pers_info_type_df) * 0.9), :]
    pers_info_type_test_df = pers_info_type_df.iloc[int(len(pers_info_type_df) * 0.9):, :]
    pers_info_type_train_df.to_csv(os.path.join(data_path, 'pers_info_type_train.csv'), index=False)
    pers_info_type_test_df.to_csv(os.path.join(data_path, 'pers_info_type_test.csv'), index=False)


get_unique_value()
