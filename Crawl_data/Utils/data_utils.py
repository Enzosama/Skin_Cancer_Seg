import csv
from model.SCDataset import SCDataset

#Check duplicate field
def is_duplicate_SCDataset(SC_title: str, seen_title: set) -> bool:
    return SC_title in seen_title

def is_complete_SCDataset(item: dict, required_keys: list) -> bool:
    return all(key in item for key in required_keys)

def save_SCDataset_to_csv(sc_datasets: list, filename: str):
    if not sc_datasets:
        print("No datasets to save.")
        return

    # Use field names from the SCDataset model
    fieldnames = SCDataset.model_fields.keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        rows_to_write = []
        for item in sc_datasets:
            if isinstance(item, SCDataset): 
                 try:
                      rows_to_write.append(item.model_dump()) # Pydantic V2
                 except AttributeError:
                      try:
                           rows_to_write.append(item.dict()) # Pydantic V1
                      except AttributeError:
                           print(f"Error: Item is not a recognized SCDataset object or dict: {item}")
                           continue # Bỏ qua item lỗi
            elif isinstance(item, dict): # Nếu list chứa sẵn dict
                 rows_to_write.append(item)
            else:
                 print(f"Error: Item is not a recognized SCDataset object or dict: {item}")
                 continue # Bỏ qua item lỗi
        # Write rows to CSV
        writer.writerows(rows_to_write)

    # Sửa lỗi: Sử dụng len(sc_datasets) và mô tả đúng đối tượng
    print(f"Saved {len(rows_to_write)} SCDatasets to '{filename}'.") # In số lượng thực sự đã ghi