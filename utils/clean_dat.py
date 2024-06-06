import os
import glob

directory_path = "."

dat_files = glob.glob(os.path.join(directory_path, "*.dat"))

if dat_files:
    for file_path in dat_files:
        try:
            os.remove(file_path)
            print(f"Файл '{file_path}' был успешно удален.")
        except Exception as e:
            print(f"Ошибка при удалении файла '{file_path}': {e}")
else:
    print("Файлы .dat не найдены.")