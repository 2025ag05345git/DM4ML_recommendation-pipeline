# Feature catalog

Every aggregate is derived from the training split only. Tables are keyed by
`user_id`, `item_id`, or `interaction_id` and can be reused in training and
offline prediction.

| Entity | Feature | Type | Definition | Default/null handling |
|---|---|---|---|---|
| User | `age` | integer | Synthetic profile age | Median imputation |
| User | `occupation` | category | Lowercase normalized profile occupation | `unknown` |
| User | `preferred_genre` | category | Synthetic preference used to generate behavior | Generated category |
| User | `user_activity_count` | integer | Number of training interactions | 0 |
| User | `user_average_rating` | float | Mean training rating | 0.0 |
| User | `user_preferred_genre` | category | Most common genre in training history | `Unknown` |
| User | `days_since_last_interaction` | integer | Days from last user event to training cutoff | -1 without history |
| Movie | `genre` | category | Synthetic movie genre | `Unknown` |
| Movie | `release_year` | integer | Synthetic year (1920–2025) | Nullable integer |
| Movie | `language`, `language_code` | category | Generated language marker (`English`, `en`) | English |
| Movie | `item_popularity` | integer | Training interaction count | 0 |
| Movie | `item_average_rating` | float | Mean training rating | 0.0 |
| Movie | `days_since_last_interaction` | integer | Days from movie's latest train event to cutoff | -1 without history |
| Interaction | `rating` | float | Explicit synthetic 1–5 user rating | Invalid/missing quarantined |
| Interaction | `timestamp` | UTC timestamp | Synthetic event time | Invalid values quarantined |
| Interaction | `days_since_last_interaction` | integer | Days from event to train cutoff | Computed |
| Interaction | `age`, `occupation` | numeric/category | Joined user profile fields | Defaults above |
| Interaction | `genre`, `release_year`, `language` | category/numeric | Joined movie fields | English catalog metadata |
