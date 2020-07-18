import os
import json
import pandas as pd
from sklearn.utils import shuffle

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(cur_path), 'data')
attribute_path = os.path.join(data_path, 'attribute')
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

    # Extract Choice Scope
    extract_attribute([first_party_df, third_party_df, user_choice_df], "Choice Scope", "choice_scope")

    # Extract Choice Type
    extract_attribute([first_party_df, third_party_df, user_choice_df], "Choice Type", "choice_type")

    # Extract User Type
    extract_attribute([first_party_df, third_party_df, user_choice_df, user_access_df], "User Type", "user_type")

    # Extract Personal Information Type
    extract_attribute([first_party_df, third_party_df, user_choice_df, data_retention_df], "Personal Information Type", "pers_info_type")

    # Extract Purpose
    extract_attribute([first_party_df, third_party_df, user_choice_df], "Purpose", "purpose")

    # Extract Identifiability
    extract_attribute([first_party_df, third_party_df], "Identifiability", "identifiability")

    # Extract Does/Does Not
    extract_attribute([first_party_df, third_party_df], "Does/Does Not", "does_does_not")

    # Extract Collection Mode
    extract_attribute([first_party_df], "Collection Mode", "collection_mode")

    # Extract Action First-Party
    extract_attribute([first_party_df], "Action First-Party", "action_first_party")

    # Extract Action Third Party
    extract_attribute([third_party_df], "Action Third Party", "action_third_party")

    # Extract Third Party Entity
    extract_attribute([third_party_df], "Third Party Entity", "third_party_entity")

    # Extract Access Type
    extract_attribute([user_access_df], "Access Type", "access_type")

    # Extract Access Scope
    extract_attribute([user_access_df], "Access Scope", "access_scope")

    # Extract Retention Period
    extract_attribute([data_retention_df], "Retention Period", "retention_period")

    # Extract Retention Purpose
    extract_attribute([data_retention_df], "Retention Purpose", "retention_purpose")

    # Extract Security Measure
    extract_attribute([data_security_df], "Security Measure", "security_measure")

    # Extract Change Type
    extract_attribute([policy_change_df], "Change Type", "change_type")

    # Extract User Choice
    extract_attribute([policy_change_df], "User Choice", "user_choice")

    # Extract Notification Type
    extract_attribute([policy_change_df], "Notification Type", "change_type")

    # Extract Do Not Track policy
    extract_attribute([do_not_track_df], "Do Not Track policy", "change_type")

    # Extract Audience Type
    extract_attribute([audience_df], "Audience Type", "change_type")

    # Extract Other Type
    extract_attribute([other_df], "Other Type", "change_type")


def extract_attribute(category_df_list, attribute_name, file_name):
    cols = ['segment_content', 'attributes']
    attribute_df = pd.DataFrame(columns=cols)
    attributes = ['startIndexInSegment', 'endIndexInSegment', 'selectedText', 'value']
    for df in category_df_list:
        seg_attributes = {}
        for row in df.iterrows():
            seg_content = row[1][0]
            all_attributes = json.loads(row[1][2])
            target_attr = all_attributes.get(attribute_name)
            attr_val = [target_attr.get(attr, '') for attr in attributes]
            if seg_content not in seg_attributes:
                seg_attributes[seg_content] = [attr_val]
            else:
                seg_attributes[seg_content].append(attr_val)

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
            attr_str = '║'.join(['☀'.join([str(m) for m in merg]) for merg in merged])
            if attr_str:
                attribute_df.loc[attribute_df.shape[0] + 1] = [seg] + [attr_str]
    attribute_df.to_csv(os.path.join(attribute_path, file_name + '.csv'), index=False)
    shuffle(attribute_df)
    attribute_train_df = attribute_df.iloc[:int(len(attribute_df) * 0.9), :]
    attribute_test_df = attribute_df.iloc[int(len(attribute_df) * 0.9):, :]
    attribute_train_df.to_csv(os.path.join(attribute_path, file_name + '_train.csv'), index=False)
    attribute_test_df.to_csv(os.path.join(attribute_path, file_name + '_test.csv'), index=False)
    print('Finished extraction for attribute: %s ' % attribute_name)

get_unique_value()
