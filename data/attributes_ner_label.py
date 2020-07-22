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
    "Third party use": ["B-TPU", "I-TPU"],
    "Unspecified": ["B-UNS", "I_UNS"]
}
cho_scope_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-BOTH", "I-BOTH", "B-COL", "I-COL", "B-USE", "I-USE", "B-FPC",
                        "I-FPC", "B-FPU", "I-FPU", "B-TPS", "I-TPS", "B-TPU", "I-TPU", "B-UNS", "I_UNS"]


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
cho_type_ner_dict = {
    "Browser/device privacy controls": ["B-BRO", "I-BRO"],
    "Dont use service/feature": ["B-DON", "I-DON"],
    "First-party privacy controls": ["B-FIR", "I-FIR"],
    "Opt-in": ["B-OPI", "I-OPI"],
    "Opt-out link": ["B-OPU", "I-OPU"],
    "Opt-out via contacting company": ["B-OPUC", "I-OPUC"],
    "Third-party privacy controls": ["B-THD", "I-THD"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
cho_type_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-BRO", "I-BRO", "B-DON", "I-DON", "B-FIR", "I-FIR", "B-OPI",
                       "I-OPI", "B-OPU", "I-OPU", "B-OPUC", "I-OPUC", "B-THD", "I-THD", "B-OTH", "I-OTH", "B-UNS",
                       "I-UNS"]


## User Type
# * User with account
# * User without account
# * not-selected
# * Other
# * Unspecified
user_type_ner_dict = {
    "User with account": ["B-UWA", "I-UWA"],
    "User without account": ["B-UWO", "I-UWO"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
user_type_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-UWA", "I-UWA", "B-UWO", "I-UWO", "B-OTH", "I-OTH", "B-UNS",
                        "I-UNS"]


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


## Purpose
# * Additional service/feature
# * Advertising
# * Analytics/Research
# * Basic service/feature
# * Legal requirement
# * Marketing
# * Merger/Acquisition
# * Personalization/Customization
# * Service operation and security
# * Other
# * Unspecified
purpose_ner_dict = {
    "Additional service/feature": ["B-ADD", "I-ADD"],
    "Advertising": ["B-ADV", "I-ADV"],
    "Analytics/Research": ["B_ANA", "I-ANA"],
    "Basic service/feature": ["B-BAS", "I-BAS"],
    "Legal requirement": ["B-LEG", "I-LEG"],
    "Marketing": ["B-MAR", "I-MAR"],
    "Merger/Acquisition": ["B-MER", "I-MER"],
    "Personalization/Customization": ["B-PER", "I-PER"],
    "Service operation and security": ["B-SER", "I-SER"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
purpose_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-ADD", "I-ADD", "B-ADV", "I-ADV", "B_ANA", "I-ANA", "B-BAS", "I-BAS", "B-LEG", "I-LEG",
                      "B-MAR", "I-MAR", "B-MER", "I-MER", "B-PER", "I-PER", "B-SER", "I-SER", "B-OTH", "I-OTH",
                      "B-UNS", "I-UNS"]

## Identifiability
# * Aggregated or anonymized
# * Identifiable
# * not-selected
# * Other
# * Unspecified
identifiability_ner_dict = {
    "Aggregated or anonymized": ["B-AGG", "I-AGG"],
    "Identifiable": ["B-IDE", "I-IDE"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
identifiability_ner_labels = ["[PAD]", "O", "X", "[CLS]", "B-AGG", "I-AGG", "B-IDE", "I-IDE", "B-OTH", "I-OTH",
                              "B-UNS", "I-UNS"]


## Does/Does Not
# * Does
# * Does Not
does_dose_not_ner_dict = {
    "Does": ["B-DOES", "I-DOES"],
    "Does Not": ["B-DNT", "I-DNT"]
}
does_dose_not_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-DOES", "I-DOES", "B-DNT", "I-DNT"]


## Collection Mode
# * Implicit
# * Explicit
# * not-selected
# * Unspecified
collection_mode_ner_dict = {
    "Implicit": ["B-IMP", "I-IMP"],
    "Explicit": ["B-EXP", "I-EXP"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
collection_mode_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-IMP", "I-IMP", "B-EXP", "I-EXP", "B-UNS", "I-UNS"]


## Action First-Party
# * Collect from user on other websites
# * Collect in mobile app
# * Collect on mobile website
# * Collect on website
# * Receive from other service/third-party (unnamed)
# * Receive from other service/third-party (named)
# * Receive from other parts of company/affiliates
# * Track user on other websites
# * Other
# * Unspecified
action_first_party_ner_dict = {
    "Collect from user on other websites": ["B-CFU", "I-CFU"],
    "Collect in mobile app": ["B-CMA", "I-CMA"],
    "Collect on mobile website": ["B-CMW", "I-CMW"],
    "Collect on website": ["B-COW", "I-COW"],
    "Receive from other service/third-party (unnamed)": ["B-RFU", "I-RFU"],
    "Receive from other service/third-party (named)": ["B-RFN", "I-RFN"],
    "Receive from other parts of company/affiliates": ["B-RFP", "I-RFP"],
    "Track user on other websites": ["B-TUO", "I-TUO"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
action_first_party_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-CFU", "I-CFU", "B-CMA", "I-CMA", "B-CMW", "I-CMW",
                                "B-COW", "I-COW", "B-RFU", "I-RFU", "B-RFN", "I-RFN", "B-RFP", "I-RFP", "B-TUO",
                                "I-TUO", "B-OTH", "I-OTH", "B-UNS", "I-UNS"]


## Third Party Entity
# * Named third party
# * Unnamed third party
# * Other part of company/affiliate
# * Public
# * Other users
# * Other
# * Unspecified
third_party_ner_dict = {
    "Named third party": ["B-NTP", "I-NTP"],
    "Unnamed third party": ["B-UTP", "I-UTP"],
    "Other part of company/affiliate": ["B-OPC", "I-OPC"],
    "Public": ["B-PUB", "I-PUB"],
    "Other users": ["B-OTU", "I-OTU"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
third_party_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-NTP", "I-NTP", "B-UTP", "I-UTP", "B-OPC", "I-OPC", "B-OPC",
                         "I-OPC", "B-PUB", "I-PUB", "B-OTU", "I-OTU", "B-OTH", "I-OTH", "B-UNS", "I-UNS"]


## Access Type
# * None
# * View
# * Export
# * Edit information
# * Deactivate account
# * Delete account (full)
# * Delete account (partial)
# * Other
# * Unspecified
access_type_ner_dict = {
    "None": ["B-NONE", "I-NONE"],
    "View": ["B-VIEW", "I-VIEW"],
    "Export": ["B-EXP", "I-EXP"],
    "Edit information": ["B-EDT", "I-EDT"],
    "Deactivate account": ["B-DEA", "I-DEA"],
    "Delete account (full)": ["B-DELF", "I-DELF"],
    "Delete account (partial)": ["B-DELP", "I-DELP"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
access_type_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-NONE", "I-NONE", "B-VIEW", "I-VIEW", "B-EXP", "I-EXP",
                         "B-EDT", "I-EDT", "B-DEA", "I-DEA", "B-DELF", "I-DELF", "B-DELP", "I-DELP", "B-OTH",
                         "I-OTH", "B-UNS", "I-UNS"]


## Access Scope
# * User account data
# * Profile data
# * Transactional data
# * Other data about user
# * Other
# * Unspecified
access_scope_ner_dict = {
    "User account data": ["B-UAD", "I-UAD"],
    "Profile data": ["B-PRO", "I-PRO"],
    "Transactional data": ["B-TRA", "I-TRA"],
    "Other data about user": ["B-OTA", "I-OTA"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
access_scope_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-UAD", "I-UAD", "B-PRO", "I-PRO", "B-TRA", "I-TRA", "B-OTA",
                          "I-OTA", "B-OTH", "I-OTH", "B-UNS", "I-UNS"]


## Retention Period
# * Indefinitely
# * Limited
# * Stated Period
# * Other
# * Unspecified
retention_period_ner_dict = {
    "Indefinitely": ["B-IND", "I-IND"],
    "Limited": ["B-LIM", "I-LIM"],
    "Stated Period": ["B-STA", "I-STA"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
retention_period_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-IND", "I-IND", "B-LIM", "I-LIM", "B-STA", "I-STA", "B-OTH",
                              "I-OTH", "B-UNS", "I-UNS"]


## Retention Purpose
# * Marketing
# * Advertising
# * Service operation and security
# * Analytics/Research
# * Perform service
# * Legal requirement
# * Other
# * Unspecified
retention_purpose_ner_dict = {
    "Marketing": ["B-MAR", "I-MAR"],
    "Advertising": ["B-ADV", "I-ADV"],
    "Service operation and security": ["B-SER", "I-SER"],
    "Analytics/Research": ["B-ANA", "I-ANA"],
    "Perform service": ["B-PER", "I-PER"],
    "Legal requirement": ["B-LEG", "I-LEG"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
retention_purpose_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-MAR", "I-MAR", "B-ADV", "I-ADV", "B-SER", "I-SER",
                               "B-ANA", "I-ANA", "B-PER", "I-PER", "B-LEG", "I-LEG", "B-OTH", "I-OTH", "B-UNS",
                               "I-UNS"]


## Security Measure
# * Data access limitation
# * Generic
# * Privacy review/audit
# * Privacy/Security program
# * Privacy training
# * Secure data storage
# * Secure data transfer
# * Secure user authentication
# * Other
# * Unspecified
security_measure_ner_dict = {
    "Data access limitation": ["B-DAL", "I-DAL"],
    "Generic": ["B-GEN", "I-GEN"],
    "Privacy review/audit": ["B-PRR", "I-PRR"],
    "Privacy/Security program": ["B-PRS", "I-PRS"],
    "Privacy training": ["B-PRT", "I-PRT"],
    "Secure data storage": ["B-SDS", "I-SDS"],
    "Secure data transfer": ["B-SDT", "I-SDT"],
    "Secure user authentication": ["B-SUA", "I-SUA"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
security_measure_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-DAL", "I-DAL", "B-GEN", "I-GEN", "B-PRR", "I-PRR", "B-PRS",
                              "I-PRS", "B-PRT", "I-PRT", "B-SDS", "I-SDS", "B-SDT", "I-SDT", "B-SUA", "I-SUA", "B-OTH",
                              "I-OTH", "B-UNS", "I-UNS"]


## Change Type
# * Privacy relevant change
# * Non-privacy relevant change
# * In case of merger or acquisition
# * Other
# * Unspecified
change_type_ner_dict = {
    "Privacy relevant change": ["B-PRI", "I-PRI"],
    "Non-privacy relevant change": ["B-NRC", "I-NRC"],
    "In case of merger or acquisition": ["B-INC", "I-INC"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
change_type_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-PRI", "I-PRI", "B-NRC", "I-NRC", "B-INC", "I-INC", "B-OTH",
                         "I-OTH", "B-UNS", "I-UNS"]


## User Choice
# * User participation
# * Opt-in
# * Opt-out
# * None
# * Other
# * Unspecified
user_choice_ner_dict = {
    "User participation": ["B-USER", "I-USER"],
    "Opt-in": ["B-OPI", "I-OPI"],
    "Opt-out": ["B-OPO", "I-OPO"],
    "None": ["B-NONE", "I-NONE"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
user_choice_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-USER", "I-USER", "B-OPI", "I-OPI", "B-OPO", "I-OPO", "B-NONE",
                         "I-NONE", "B-OTH", "I-OTH", "B-UNS", "I-UNS"]


## Notification Type
# * General notice in privacy policy
# * General notice on website
# * Personal notice
# * No notification
# * Other
# * Unspecified
notif_type_ner_dict = {
    "General notice in privacy policy": ["B-GNI", "I-GNI"],
    "General notice on website": ["B-GNO", "I-GNO"],
    "Personal notice": ["B-PER", "I-PER"],
    "No notification": ["B-NON", "I-NON"],
    "Other": ["B-OTH", "I-OTH"],
    "Unspecified": ["B-UNS", "I-UNS"]
}
notif_type_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-GNI", "I-GNI", "B-GNO", "I-GNO", "B-PER", "I-PER", "B-NON",
                        "I-NON", "B-OTH", "I-OTH", "B-UNS", "I-UNS"]


## Do Not Track policy
# * Honored
# * Not honored
# * Mentioned, but unclear if honored
# * Other
do_not_track_ner_dict = {
    "Honored": ["B-HON", "I-HON"],
    "Not honored": ["B-NHON", "I-NHON"],
    "Mentioned, but unclear if honored": ["B-MEN", "I-MEN"],
    "Other": ["B-OTH", "I-OTH"]
}
do_not_track_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-HON", "I-HON", "B-NHON", "I-NHON", "B-MEN", "I-MEN", "B-OTH",
                          "I-OTH"]


## Audience Type
# * Californians
# * Children
# * Europeans
# * Citizens from other countries
# * Other
audience_type_ner_dict = {
    "Californians": ["B-CAL", "I-CAL"],
    "Children": ["B-CHI", "I-CHI"],
    "Europeans": ["B-EUR", "I-EUR"],
    "Citizens from other countries": ["B-CIT", "I-CIT"],
    "Other": ["B-OTH", "I-OTH"]
}
audience_type_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-CAL", "I-CAL", "B-CHI", "I-CHI", "B-EUR", "I-EUR", "B-CIT",
                           "I-CIT", "B-OTH", "I-OTH"]


## Other Type
# * Introductory/Generic
# * Privacy contact information
# * Practice not covered
# * Other
other_type_ner_dict = {
    "Introductory/Generic": ["B-INT", "I-INT"],
    "Privacy contact information": ["B-PRI", "I-PRI"],
    "Practice not covered": ["B-PRA", "I-PRA"],
    "Other": ["B-OTH", "I-OTH"]
}
other_type_ner_label = ["[PAD]", "O", "X", "[CLS]", "B-INT", "I-INT", "B-PRI", "I-PRI", "B-PRA", "I-PRA", "B-OTH",
                        "I-OTH"]


attribute_infos = {
    "Choice Scope": {
        "dict": cho_scope_ner_dict,
        "label": cho_scope_ner_labels,
        "file_prefix": "choice_scope"
    },
    "Choice Type": {
        "dict": cho_type_ner_dict,
        "label": cho_type_ner_labels,
        "file_prefix": "choice_type"
    },
    "User Type": {
        "dict": user_type_ner_dict,
        "label": user_type_ner_labels,
        "file_prefix": "user_type"
    },
    "Personal Information Type": {
        "dict": pers_info_ner_dict,
        "label": pers_info_ner_labels,
        "file_prefix": "pers_info_type"
    },
    "Purpose": {
        "dict": purpose_ner_dict,
        "label": purpose_ner_labels,
        "file_prefix": "purpose"
    },
    "Identifiability": {
        "dict": identifiability_ner_dict,
        "label": identifiability_ner_labels,
        "file_prefix": "identifiability"
    },
    "Does/Does Not": {
        "dict": does_dose_not_ner_dict,
        "label": does_dose_not_ner_label,
        "file_prefix": "does_does_not"
    },
    "Collection Mode": {
        "dict": collection_mode_ner_dict,
        "label": collection_mode_ner_label,
        "file_prefix": "collection_mode"
    },
    "Action First-Party": {
        "dict": action_first_party_ner_dict,
        "label": action_first_party_ner_label,
        "file_prefix": "action_first_party"
    },
    "Third Party Entity": {
        "dict": third_party_ner_dict,
        "label": third_party_ner_label,
        "file_prefix": "third_party_entity"
    },
    "Access Type": {
        "dict": access_type_ner_dict,
        "label": access_type_ner_label,
        "file_prefix": "access_type"
    },
    "Access Scope": {
        "dict": access_scope_ner_dict,
        "label": access_scope_ner_label,
        "file_prefix": "access_scope"
    },
    "Retention Period": {
        "dict": retention_period_ner_dict,
        "label": retention_period_ner_label,
        "file_prefix": "retention_period"
    },
    "Retention Purpose": {
        "dict": retention_purpose_ner_dict,
        "label": retention_purpose_ner_label,
        "file_prefix": "retention_purpose"
    },
    "Security Measure": {
        "dict": security_measure_ner_dict,
        "label": security_measure_ner_label,
        "file_prefix": "security_measure"
    },
    "Change Type": {
        "dict": change_type_ner_dict,
        "label": change_type_ner_label,
        "file_prefix": "change_type"
    },
    "User Choice": {
        "dict": user_choice_ner_dict,
        "label": user_choice_ner_label,
        "file_prefix": "user_choice"
    },
    "Notification Type": {
        "dict": notif_type_ner_dict,
        "label": notif_type_ner_label,
        "file_prefix": "notification_type"
    },
    "Do Not Track policy": {
        "dict": do_not_track_ner_dict,
        "label": do_not_track_ner_label,
        "file_prefix": "do_not_track"
    },
    "Audience Type": {
        "dict": audience_type_ner_dict,
        "label": audience_type_ner_label,
        "file_prefix": "audience_type"
    },
    "Other Type": {
        "dict": other_type_ner_dict,
        "label": other_type_ner_label,
        "file_prefix": "other_type"
    }
}
