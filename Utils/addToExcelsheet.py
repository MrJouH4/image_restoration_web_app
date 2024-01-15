import pandas as pd


def add_to_excelsheet(excel_file_path, epoch_number, epoch_loss, lr):
    excel_file_path = excel_file_path
    learning_rate = lr

    data = {'Epoch': [epoch_number], 'Loss': [epoch_loss], 'Learning Rate': [learning_rate]}
    df = pd.DataFrame(data)

    try:
        existing_df = pd.read_excel(excel_file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        writer.book.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        df.to_excel(writer, sheet_name='Sheet1', index=False)
