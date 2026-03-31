import pandas as pd
from joblib import dump

# ==================
# Preprocessing Data
# ==================
def processor_data(df_user, df_book, df_rating):
    # ============
    # Prepare Data
    # ============
    # ======== Prepare Rating Data =======
    # === Filter Interactions ===
    df_rating = df_rating[df_rating['Book-Rating'] > 6]

    # === Merge Tables ===
    df_rating_book = pd.merge(df_rating, df_book, on="ISBN", how="left")
    df_working = pd.merge(df_rating_book, df_user, on="User-ID", how="left")
    df_working = df_working.dropna()

    # ======== Prepare Books Data =======
    # === Item ID Encoding ===
    all_items = df_working['ISBN'].unique().tolist()
    items_to_idx = {item: idx for idx, item in enumerate(all_items)}
    df_working["book_idx"] = df_working["ISBN"].map(items_to_idx).astype(int)
    items_idx_to_id = {idx: id for idx, id in zip(df_working["book_idx"], df_working['ISBN'])}

    popular_items = df_working["book_idx"].value_counts().head(100).index.tolist()
    print(f"Popular Items: {popular_items}")


    # Item to Title
    def validate_title_column(x):
      if pd.isna(x) or x is None:
        return "unknown"
      return x

    df_working["Book-Title"] = df_working["Book-Title"].apply(validate_title_column)

    item_idx_to_title = {idx: title for idx, title in zip(df_working["book_idx"], df_working["Book-Title"])}

    # === Book-Author to idx ===
    def validate_author_column(x):
      if pd.isna(x) or x is None:
        return "unknown"
      return x

    df_working["Book-Author"] = df_working["Book-Author"].apply(validate_author_column)
    Authors = df_working['Book-Author'].unique().tolist()
    author_idx = {author: idx for idx, author in enumerate(Authors)}
    df_working["author_idx"] = df_working["Book-Author"].map(author_idx).astype(int)
    
    # === Making the Year of Production a Category Value ===
    def validate_year_column(year_production):
        if pd.isna(year_production) or year_production is None:
            return "unknown"
        elif type(year_production) == str:
            try:
                return int(year_production)
            except:
                return "unknown"
        else:
            return year_production


    def categorize_year(year):
        if year == "unknown":
            return "unknown"
        elif year < 1500:
            return "< 1500"
        elif year < 1700:
            return "1500-1699"
        elif year < 1800:
            return "1700-1799"
        elif year < 1900:
            return "1800-1899"
        elif year < 1950:
            return "1900-1949"
        elif year < 1980:
            return "1950-1979"
        elif year < 2000:
            return "1980-1999"
        elif year < 2026:
            return "2000-2025"
        else:
            return ">= 2026"

    df_working["year_production"] = df_working["Year-Of-Publication"].map(validate_year_column)
    df_working["year_production"] = df_working["year_production"].map(categorize_year)

    # === Year Encoding ===
    categorical_years = df_working["year_production"].unique().tolist()
    category_year_idx = {cat_year: idx for idx, cat_year in enumerate(categorical_years)}

    df_working["year_production_idx"] = df_working["year_production"].map(category_year_idx).astype(int)

    # === Bublisher Encoding ===
    def validate_publisher_column(x):
        if pd.isna(x) or x is None:
            return "unknown"
        return x

    df_working["Publisher"] = df_working["Publisher"].apply(validate_publisher_column)
    publisher = df_working["Publisher"].unique().tolist()
    publisher_idx = {p: idx for idx, p in enumerate(publisher)}
    df_working["publisher_idx"] = df_working["Publisher"].map(publisher_idx).astype(int)

    # === storing Images Links ===
    def safe_url(x):
        if pd.isna(x) or x is None:
            return "url-not-found"
        return x

    df_working["Image-URL-S"] = df_working["Image-URL-S"].apply(safe_url)
    df_working["Image-URL-M"] = df_working["Image-URL-M"].apply(safe_url)
    df_working["Image-URL-L"] = df_working["Image-URL-L"].apply(safe_url)

    book_idx_to_images_links = {
        idx: {
            "image_url_s": url_s,
            "image_url_m": url_m,
            "image_url_l": url_l
        } for idx, url_s, url_m, url_l in zip(df_working["book_idx"], df_working["Image-URL-S"], df_working["Image-URL-M"], df_working["Image-URL-L"])
    }


    # ======== Prepare Users Data =======
    # === Users Encoding ===
    all_users = df_working["User-ID"].unique().tolist()
    user_id_to_idx = {id: idx for idx, id in enumerate(all_users)}
    df_working["user_idx"] = df_working["User-ID"].map(user_id_to_idx).astype(int)
    user_idx_to_id = {idx: id for idx, id in zip(df_working["user_idx"], df_working["User-ID"])}
    print("user_idx: ",df_working["user_idx"].dtype)
    print("User-ID: ",df_working["User-ID"].dtype)

    # === User Location Encoding ===
    def validate_location_column(x):
      if pd.isna(x) or x is None:
        return "unknown"
      return x

    df_working["Location"] = df_working["Location"].apply(validate_location_column)
    locations = df_working["Location"].unique().tolist()
    location_idx = {l: idx for idx, l in enumerate(locations)}
    df_working["location_idx"] = df_working["Location"].map(location_idx).astype(int)

    # === User Age Encoding ===
    def validate_age_column(age):
        if pd.isna(age) or age is None:
            return "unknown"
        return age

    def categorize_ages(age):
        if age == "unknown":
            return "unknown"
        elif age < 13:
            return "Child"
        elif age < 18:
            return "Teen"
        elif age < 26:
            return "Young Adult"
        elif age < 36:
            return "Adult"
        elif age < 50:
            return "Middle"
        else:
            return "+50"

    df_working["Age"] = df_working["Age"].apply(validate_age_column)
    df_working["age_category"] = df_working["Age"].apply(categorize_ages)

    # Encoding Ages
    age_categories = df_working["age_category"].unique().tolist()
    age_idx = {c: idx for idx, c in enumerate(age_categories)}
    df_working["age_idx"] = df_working["age_category"].map(age_idx).astype(int)


    df_working["user_idx"] = df_working["user_idx"].astype(int)
    df_working["book_idx"] = df_working["book_idx"].astype(int)

    final_working_data = df_working[["user_idx", "book_idx", "age_idx", "location_idx", "publisher_idx", "year_production_idx", "author_idx"]]
    print(final_working_data["user_idx"].nunique())

    # =================
    # Load Working Data
    # =================
    dump(items_to_idx, "storage/store/items_id_to_idx.pkl")
    dump(items_idx_to_id, "storage/store/items_idx_to_id.pkl")
    dump(item_idx_to_title, "storage/store/item_idx_to_title.pkl")
    dump(author_idx, "storage/store/author_idx.pkl")
    dump(category_year_idx, "storage/store/category_year_idx.pkl")
    dump(publisher_idx, "storage/store/publisher_idx.pkl")
    dump(book_idx_to_images_links, "storage/store/book_idx_to_images_links.pkl")
    dump(user_id_to_idx, "storage/store/user_id_to_idx.pkl")
    dump(user_idx_to_id, "storage/store/user_idx_to_id.pkl")
    dump(location_idx, "storage/store/location_to_idx.pkl")
    dump(age_idx, "storage/store/age_to_idx.pkl")
    dump(popular_items, "storage/store/popular_items.pkl")

    final_working_data.to_csv("data/training_data.csv")

    print("Done.")