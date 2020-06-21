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
## change-type
* privacy-relevant-change
* unspecified

## notification-type
* general-notice-in-privacy-policy
* general-notice-on-website
* personal-notice
* unspecified

## do-not-track-policy
* honored
* not-honored

## security-measure
* data-access-limitation
* generic
* privacy-review-audit
* privacy-security-program
* secure-data-storage
* secure-data-transfer
* secure-user-authentication

## personal-information-type
* computer-information
* contact
* cookies-and-tracking-elements
* demographic
* financial
* generic-personal-information
* health
* ip-address-and-device-ids
* location
* personal-identifier
* social-media-data
* survey-data
* unspecified
* user-online-activities
* user-profile

## purpose
* additional-service-feature
* advertising
* analytics-research
* basic-service-feature
* legal-requirement
* marketing
* merger-acquisition
* personalization-customization
* service-operation-and-security
* unspecified

## choice-type
* browser-device-privacy-controls
* dont-use-service-feature
* first-party-privacy-controls
* opt-in
* opt-out-link
* opt-out-via-contacting-company
* third-party-privacy-controls
* unspecified

## third-party-entity
* collect-on-first-party-website-app
* receive-shared-with
* see
* track-on-first-party-website-app
* unspecified

## access-type
* edit-information
* unspecified
* view

## audience-type
* californians
* children
* europeans

## choice-scope
* both
* collection
* first-party-collection
* first-party-use
* third-party-sharing-collection
* third-party-use
* unspecified
* use

## action-first-party
* collect-in-mobile-app
* collect-on-mobile-website
* collect-on-website
* unspecified

## does-does-not
* does
* does-not

## retention-period
* indefinitely
* limited
* stated-period
* unspecified

## identifiability
* aggregated-or-anonymized
* identifiable
* unspecified