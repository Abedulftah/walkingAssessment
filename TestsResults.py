import openpyxl


def get_row(sheet, row_name, session_num):
    column_name = 'Number'

    column_index = None
    for column in sheet.iter_cols():
        if column[0].value == column_name:
            column_index = column[0].column
            break

    row_index = 1
    for patient in sheet.iter_rows(min_col=column_index, max_col=column_index):
        if patient[0].value == row_name:
            return row_index
        row_index += 1

    return -1


def get_start_time(row_name, session_num):
    workbook = openpyxl.load_workbook('SPSS.xlsx')
    sheet = workbook['RESULTS']

    row_index = get_row(sheet, row_name, session_num)

    column_name = '4MWTspeed_' + session_num
    column_index = None
    for column in sheet.iter_cols():
        if column[0].value == column_name:
            column_index = column[0].column
            break

    cell = sheet.cell(row=row_index, column=column_index)
    cell_val = cell.value

    return cell_val


def save_evaluation(path, walking_speed):
    workbook = openpyxl.load_workbook('everything.xlsx')

    # Select the worksheet
    worksheet = workbook.active

    # Get the existing column
    nameOf = path.split('/')
    nameOf = nameOf[len(nameOf) - 1].split('_')
    nameOf[1] = nameOf[1].split('.')[0]
    print(nameOf)
    for i in range(1, worksheet.max_row):
        if worksheet.cell(row=i, column=1).value == int(nameOf[0]) and \
                worksheet.cell(row=i, column=2).value == nameOf[1]:
            cell_value = worksheet.cell(row=i, column=worksheet.max_column).value
            if cell_value is not None or cell_value != "":
                worksheet.cell(row=i, column=worksheet.max_column, value=walking_speed)

    # Save the modified Excel file
    workbook.save('everything.xlsx')

