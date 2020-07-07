# Website resource for privacy policy
* Selected from google query result based on relevance --> 1799 unique websites
* Organized into 15 sectors, e.g., Arts, Shopping, Business, News
* Excluded the "World" sector and limited the "Regional" sector to the "U.S." subsector in order to ensure that
all privacy policies are subject to the same legal and regulatory requirements.
* Final dataset is 115 privacy policies across 15 sectors

# Annotation Schema
The final annotation schema consists of 10 data practice categories:
* First Party Collection/Use: how and why a service provider collects user information.
* Third Party Sharing/Collection: how user information may be shared with or collected by third parties.
* User Choice/Control: choices and control options available to users.
* User Access, Edit, & Deletion: if and how users may access, edit, or delete their information.
* Data Retention: how long user information is stored.
* Data Security: how user information is protected.
* Policy Change: if and how users will be informed about changes to the privacy policy.
* Do Not Track: if and how Do Not Track signals for online tracking and advertising are honored.
* International & Specific Audiences: practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents).
* Other: additional sub-labels for introductory or general text, contact information, and practices not covered by the other categories.


## What is Do Not Track?
什么是跟踪？
跟踪是指网站、第三方内容提供商、广告商和其他各方了解你如何与站点进行交互的方法。这可能包括跟踪你访问的页面、单击的链接、购买或查看的产品。这有助于这些站点提供个性化内容，如广告或推荐，但这也意味着你的浏览活动会被收集起来，并常常与其他公司共享。

使用“Do Not Track”以帮助保护你的隐私
打开“Do Not Track”功能后，Internet Explorer 将向所访问的站点以及在这些站点中托管其内容的第三方发送“Do Not Track”请求。“Do Not Track”请求可让这些站点和内容提供商知道你不想让自己的浏览活动被跟踪。

注意
向站点发送“Do Not Track”请求不能保证隐私保护。站点可能选择尊重此请求，也可能继续从事你认为是跟踪的活动，即使你表达了这种意向。这取决于各个站点的隐私做法。


# Detailed Classification results
* These information block is derived from training data

|First Party Collection/Use|Third Party Sharing/Collection|User Choice/Control|User Access, Edit and Deletion|Data Retention|Data Security|Policy Change|Do Not Track|International and Specific Audiences|Other|
|  :----:                  |               :----:         |         :----:    |   :----:                     |  :----:      |  :----:     |  :----:     |  :----:    |  :----:                            |:----:|
| Choice Scope | Choice Scope | Choice Scope  | | | | | | | |
| Choice Type | Choice Type | Choice Type | | | | | | | |
| User Type | User Type | User Type | User Type | | | | | | |
| Personal Information Type | Personal Information Type | Personal Information Type | | Personal Information Type | | | | | |
| Purpose | Purpose | Purpose | | | | | | | |
| Identifiability | Identifiability | | | | | | | | |
| Does/Does Not | Does/Does Not | | | | | | | | |
| Collection Mode | | | | | | | | | |
| Action First-Party | | | | | | | | | |
| | Action Third Party | | | | | | | | |
| | Third Party Entity | | | | | | | | |
| | | | Access Type | | | | | | |
| | | | Access Scope | | | | | | |
| | | | | Retention Period | | | | | |
| | | | | Retention Purpose | | | | | |
| | | | | | Security Measure | | | | |
| | | | | | | Change Type | | | |
| | | | | | | User Choice | | | |
| | | | | | | Notification Type | | | |
| | | | | | | | Do Not Track policy | | |
| | | | | | | | | Audience Type | |
| | | | | | | | | | Other Type |

## Choice Scope
* Both  (only appears in first_party, third_party)
* Collection  (only appears in first_party, third_party)
* Use  (only appeared in first_party, third_party)
* not-selected  (only appears in first_party, third_party)
* First party collection  (only appears in user_choice)
* First party use  (only appears in user_choice)
* Third party sharing/collection  (only appears in user_choice)
* Third party use  (only appears in user_choice)
* Unspecified


## Choice Type
* Browser/device privacy controls
* Dont use service/feature
* First-party privacy controls
* Opt-in
* Opt-out link
* Opt-out via contacting company
* Third-party privacy controls
* Other
* not-selected  (only appears in first_party, third_party)
* Unspecified

## User Type
* User with account
* User without account
* not-selected
* Other
* Unspecified

## Personal Information Type
* Computer information
* Contact
* Cookies and tracking elements
* Demographic
* Financial
* Generic personal information
* Health
* IP address and device IDs
* Location 
* Personal identifier
* Social media data  (only appears in user_choice, first_party)
* Survey data
* User online activities
* User profile
* Other 
* Unspecified 

## Purpose
* Additional service/feature
* Advertising   
* Analytics/Research
* Basic service/feature
* Legal requirement
* Marketing  
* Merger/Acquisition
* Personalization/Customization
* Service Operation and Security
* Other  
* Unspecified  

## Identifiability
* Aggregated or anonymized
* Identifiable
* not-selected
* Other
* Unspecified

## Does/Does Not
* Does
* Does Not

## Collection Mode
* Implicit
* Explicit
* not-selected
* Unspecified

## Action First-Party
* Collect from user on other websites
* Collect in mobile app
* Collect on mobile website
* Collect on website
* Receive from other service/third-party (unnamed)
* Receive from other service/third-party (named)
* Receive from other parts of company/affiliates
* Track user on other websites
* Other
* Unspecified

## Action Third Party
* Collect on first party website/app
* Track on first party website/app
* Receive/Shared with
* See
* Other
* Unspecified

## Third Party Entity
* Named third party
* Unnamed third party
* Other part of company/affiliate
* Public
* Other users
* Other
* Unspecified

## Access Type
* None
* View
* Export
* Edit information
* Deactivate account
* Delete account (full)
* Delete account (partial)
* Other
* Unspecified

## Access Scope
* User account data
* Profile data
* Transactional data
* Other data about user
* Other
* Unspecified

## Retention Period
* Indefinitely
* Limited
* Stated Period
* Other
* Unspecified

## Retention Purpose
* Marketing
* Advertising
* Service operation and security
* Analytics/Research
* Perform service
* Legal requirement
* Other
* Unspecified

## Security Measure
* Data access limitation
* Generic
* Privacy review/audit
* Privacy/Security program
* Privacy training
* Secure data storage
* Secure data transfer
* Secure user authentication
* Other
* Unspecified

## Change Type
* Privacy relevant change
* Non-privacy relevant change
* In case of merger or acquisition
* Other
* Unspecified

## User Choice
* User participation
* Opt-in
* Opt-out
* None
* Other
* Unspecified

## Notification Type
* General notice in privacy policy
* General notice on website
* Personal notice
* No notification
* Other
* Unspecified

## Do Not Track policy
* Honored
* Not honored
* Mentioned, but unclear if honored
* Other

## Audience Type
* Californians
* Children
* Europeans
* Citizens from other countries
* Other

## Other Type
* Introductory/Generic
* Privacy contact information
* Practice not covered
* Other

# How to create a data practice
For each segment, an annotator may label zero or more data practices from each category. To create a data practice, 
an annotator first selects a practice category and then specifies values and text spans for each of its attributes.

The annotation tool required the selection of a text span for mandatory attributes, but did not require a text-based 
justification for optional attributes or attributes marked as “Unspecified”.