import pandas as pd
import glob
import os

# CSV 파일들이 들어있는 폴더 경로
folder_path = "./csv_files"  # 👉 본인 CSV 파일들이 있는 폴더로 변경하세요

# 폴더 내 모든 CSV 파일 불러오기
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 여러 CSV를 하나의 DataFrame으로 합치기
df_list = [pd.read_csv(file) for file in all_files]
merged_df = pd.concat(df_list, ignore_index=True)

# 합쳐진 CSV를 저장할 파일명
output_file = "uguide_data.csv"
merged_df.to_csv(output_file, index=False)

# 총 데이터 수
total_rows = len(merged_df)

# label별 개수
label_counts = merged_df["label"].value_counts()

print(f"총 데이터 수: {total_rows}")
print("label 분포:")
print(label_counts)
