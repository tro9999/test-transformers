from datetime import date, timedelta

def get_time_period_dates(current_date):
    # Calculate ISO 8601 date for yesterday
    yesterday = current_date - timedelta(days=1)

    # Calculate ISO 8601 date for today (current date)
    today = current_date

    # Calculate ISO 8601 date for tomorrow
    tomorrow = current_date + timedelta(days=1)

    # Calculate ISO 8601 date for last night
    last_night = current_date - timedelta(days=1)

    # Calculate ISO 8601 date for this morning
    this_morning = current_date
    
    # Calculate ISO 8601 date for this week
    this_monday = today - timedelta(days=today.weekday())
    this_saturday = this_monday + timedelta(days=5)
    this_sunday = this_monday + timedelta(days=6)
    this_week = f'{this_monday.isoformat()} - {this_sunday.isoformat()}'
    
    # Calculate ISO 8601 date for last week
    #last_monday = today - timedelta(days=today.weekday())
    #last_saturday = last_monday - timedelta(days=5)
    #last_sunday = last_monday - timedelta(days=1)
    #last_week = f'{last_monday.isoformat()} - {last_sunday.isoformat()}'
    
    
    # Find the start date of the current week (Monday)
    start_of_current_week = current_date - timedelta(days=current_date.weekday())

    # Calculate the start date of the previous week
    start_of_previous_week = start_of_current_week - timedelta(weeks=1)

    # Calculate the end date of the previous week
    end_of_previous_week = start_of_previous_week + timedelta(days=6)
    
    last_week = f'{start_of_previous_week.isoformat()} - {end_of_previous_week.isoformat()}'


    # Find the end date of the current week (Sunday)
    end_of_current_week = start_of_current_week + timedelta(days=6)

    # Find the start date of the last weekend (Saturday)
    last_saturday = end_of_current_week - timedelta(days=1)

    # Find the end date of the last weekend (Sunday)
    last_sunday = end_of_current_week

    # Format the start and end dates of the last weekend in ISO 8601 format
    #iso_start_date = start_of_last_weekend.isoformat()
    #iso_end_date = end_of_last_weekend.isoformat()

    
    
    # Calculate ISO 8601 date for next week
    next_monday = this_monday + timedelta(weeks=1)
    next_saturday = next_monday + timedelta(days=5)
    next_sunday = next_monday + timedelta(days=6)
    next_week = f'{next_monday.isoformat()} - {next_sunday.isoformat()}'

    # Calculate ISO 8601 date for this month
    first_day_of_month = date(today.year, today.month, 1)
    last_day_of_month = date(today.year, today.month + 1, 1) - timedelta(days=1)
    this_month = f'{first_day_of_month.isoformat()} - {last_day_of_month.isoformat()}'

    # Calculate ISO 8601 date for last month
    first_day_of_last_month = date(today.year, today.month - 1, 1)
    last_day_of_last_month = date(today.year, today.month, 1) - timedelta(days=1)
    last_month = f'{first_day_of_last_month.isoformat()} - {last_day_of_last_month.isoformat()}'

    # Calculate ISO 8601 date for next month
    first_day_of_next_month = date(today.year, today.month + 1, 1)
    last_day_of_next_month = date(today.year, today.month + 2, 1) - timedelta(days=1)
    next_month = f'{first_day_of_next_month.isoformat()} - {last_day_of_next_month.isoformat()}'

    # Calculate ISO 8601 date for this year
    first_day_of_year = date(today.year, 1, 1)
    last_day_of_year = date(today.year, 12, 31)
    this_year = f'{first_day_of_year.isoformat()} - {last_day_of_year.isoformat()}'

    # Calculate ISO 8601 date for last year
    first_day_of_last_year = date(today.year - 1, 1, 1)
    last_day_of_last_year = date(today.year - 1, 12, 31)
    last_year = f'{first_day_of_last_year.isoformat()} - {last_day_of_last_year.isoformat()}'

    # Calculate ISO 8601 date for next year
    first_day_of_next_year = date(today.year + 1, 1, 1)
    last_day_of_next_year = date(today.year + 1, 12, 31)
    next_year = f'{first_day_of_next_year.isoformat()} - {last_day_of_next_year.isoformat()}'

    

    # Calculate ISO 8601 date for first quarter
    first_quarter_start = date(date.today().year, 1, 1)
    first_quarter_end = date(date.today().year, 3, 31)

    # Calculate ISO 8601 date for second quarter
    second_quarter_start = date(date.today().year, 4, 1)
    second_quarter_end = date(date.today().year, 6, 30)

    # Calculate ISO 8601 date for third quarter
    third_quarter_start = date(date.today().year, 7, 1)
    third_quarter_end = date(date.today().year, 9, 30)

    # Calculate ISO 8601 date for fourth quarter
    fourth_quarter_start = date(date.today().year, 10, 1)
    fourth_quarter_end = date(date.today().year, 12, 31)

    return {
        'yesterday': yesterday.isoformat(),
        'today': today.isoformat(),
        'tomorrow': tomorrow.isoformat(),
        'last night': last_night.isoformat(),
        'previous night': last_night.isoformat(),
        'this morning': this_morning.isoformat(),
        'last week': last_week,
        'previous week': last_week,
        'this week': this_week,
        'next week': next_week,
        'this month': this_month,
        'last month': last_month,
        'previous month': last_month,
        'next month': next_month,
        'next year': next_year,
        'this year': this_year,
        'last year': last_year,
        'previous year': last_year,
        'last weekend': f'{last_saturday.isoformat()} - {last_sunday.isoformat()}',
        'previous weekend': f'{last_saturday.isoformat()} - {last_sunday.isoformat()}',
        'this weekend': f'{this_saturday.isoformat()} - {this_sunday.isoformat()}',
        'next weekend': f'{next_saturday.isoformat()} - {next_sunday.isoformat()}',
        'first quarter': f'{first_quarter_start.isoformat()} - {first_quarter_end.isoformat()}',
        'second quarter': f'{second_quarter_start.isoformat()} - {second_quarter_end.isoformat()}',
        'third quarter': f'{third_quarter_start.isoformat()} - {third_quarter_end.isoformat()}',
        'fourth quarter': f'{fourth_quarter_start.isoformat()} - {fourth_quarter_end.isoformat()}'
    }

# Example usage:
#current_date = date(2023, 6, 22)  # Replace with your desired current date
#result = get_time_period_dates(current_date)
#for term, iso_date in result.items():
#    print(f'{term}: {iso_date}')
