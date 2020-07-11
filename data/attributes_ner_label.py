## Choice Scope
# * Both  (only appears in first_party, third_party)
# * Collection  (only appears in first_party, third_party)
# * Use  (only appeared in first_party, third_party)
# * not-selected  (only appears in first_party, third_party)
# * First party collection  (only appears in user_choice)
# * First party use  (only appears in user_choice)
# * Third party sharing/collection  (only appears in user_choice)
# * Third party use  (only appears in user_choice)
# * Unspecified

cho_scope_ner_dict = {
    "Both": ["B-BOTH", "I-BOTH"],
    "Collection": ["B-COL", "I-COL"],
    "Use": ["B-USE", "I-USE"],
    "First party collection": ["B-FPC", "I-FPC"],
    "First party use": ["B-FPU", "I-FPU"],
    "Third party sharing/collection": ["B-TPS", "I-TPS"],
    "Third party use": ["B-TPU", "I-TPU"]
}

cho_scope_ner_labels = ["O", "X", "B-BOTH", "I-BOTH", "B-COL", "I-COL", "B-USE", "I-USE", "B-FPC", "I-FPC", "B-FPU",
                        "I-FPU", "B-TPS", "I-TPS", "B-TPU", "I-TPU"]


## Choice Type
# * Browser/device privacy controls
# * Dont use service/feature
# * First-party privacy controls
# * Opt-in
# * Opt-out link
# * Opt-out via contacting company
# * Third-party privacy controls
# * Other
# * not-selected  (only appears in first_party, third_party)
# * Unspecified



## Personal Information Type
# * Computer information
# * Contact
# * Cookies and tracking elements
# * Demographic
# * Financial
# * Generic personal information
# * Health
# * IP address and device IDs
# * Location
# * Personal identifier
# * Social media data  (only appears in user_choice, first_party)
# * Survey data
# * User online activities
# * User profile
# * Other
# * Unspecified

pers_info_ner_dict = {
    "Computer information": ["B-COM", "I-COM"],
    "Contact": ["B-CON", "I-CON"],
    "Cookies and tracking elements": ["B-COO", "I-COO"],
    "Demographic": ["B-DEM", "I-DEM"],
    "Financial": ["B-FIN", "I-FIN"],
    "Generic personal information": ["B-GEN", "I-GEN"],
    "Health": ["B-HEA", "I-HEA"],
    "IP address and device IDs": ["B-IP", "I-IP"],
    "Location": ["B-LOC", "I-LOC"],
    "Personal identifier": ["B-PER", "I-PER"],
    "Social media data": ["B-SOC", "I-SOC"],
    "Survey data": ["B-SUR", "I-SUR"],
    "User online activities": ["B-ACT", "I-ACT"],
    "User profile": ["B-PRO", "I-PRO"],
    "User Profile": ["B-PRO", "I-PRO"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
pers_info_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-COM", "I-COM", "B-CON", "I-CON", "B-COO", "I-COO", "B-DEM",
                        "I-DEM", "B-FIN", "I-FIN", "B-GEN", "I-GEN", "B-HEA", "I-HEA", "B-IP", "I-IP", "B-LOC", "I-LOC",
                        "B-PER", "I-PER", "B-SOC", "I-SOC", "B-SUR", "I-SUR", "B-ACT", "I-ACT", "B-PRO", "I-PRO",
                        "B-OTH", "I-OTH", "B-UNS", "I-UNS"]
