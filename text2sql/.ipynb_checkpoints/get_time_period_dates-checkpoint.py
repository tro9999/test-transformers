from datetime import date, timedelta

def get_time_period_dates(current_date):
    # Calculate ISO 8601 date for yesterday
    yesterday = current_date - timedelta(days=1)

    # Calculate ISO 8601 date for today (current date)
    today = current_date

    # Calculate ISO 8601 date for tomorrow
    tomorrow = current_date + timedelta(days=1)

    """
    # Calculate ISO 8601 date for last night
    last_night = current_date - timedelta(days=1)

    # Calculate ISO 8601 date for this morning
    this_morning = current_date

    # Calculate ISO 8601 date for next week
    next_week = current_date + timedelta(weeks=1)

    # Calculate ISO 8601 date for last week
    last_week = current_date - timedelta(weeks=1)

    # Calculate ISO 8601 date for this month
    this_month = date(current_date.year, current_date.month, 1)

    # Calculate ISO 8601 date for next month
    next_month_year = current_date.year if current_date.month != 12 else current_date.year + 1
    next_month_month = current_date.month + 1 if current_date.month != 12 else 1
    next_month = date(next_month_year, next_month_month, 1)

    # Calculate ISO 8601 date for last month
    last_month_year = current_date.year if current_date.month != 1 else current_date.year - 1
    last_month_month = current_date.month - 1 if current_date.month != 1 else 12
    last_month = date(last_month_year, last_month_month, 1)

    # Calculate ISO 8601 date for this year
    this_year = date(current_date.year, 1, 1)

    # Calculate ISO 8601 date for next year
    next_year = date(current_date.year + 1, 1, 1)

    # Calculate ISO 8601 date for last year
    last_year = date(current_date.year - 1, 1, 1)

    # Calculate ISO 8601 date for last weekend
    last_weekday = (current_date.weekday() + 2) % 7  # Convert Monday (0) to Sunday (6)
    last_saturday = current_date - timedelta(days=last_weekday)
    last_sunday = last_saturday - timedelta(days=1)

    # Calculate ISO 8601 date for this weekend
    this_weekday = (current_date.weekday() + 2) % 7  # Convert Monday (0) to Sunday (6)
    this_saturday = current_date + timedelta(days=(5 - this_weekday))
    this_sunday = this_saturday + timedelta(days=1)

    # Calculate ISO 8601 date for next weekend
    next_weekday = (current_date.weekday() + 2) % 7  # Convert Monday (0) to Sunday (6)
    next_saturday = current_date + timedelta(days=(12 - next_weekday))
    next_sunday = next_saturday + timedelta(days=1)

    # Calculate ISO 8601 date for last quarter
    #print(current_date)
    """
   
    """
    current_quarter = (current_date.month - 1) // 3 + 1
    last_quarter_month = ((current_quarter - 2) * 3) % 12
    last_quarter_year = current_date.year if last_quarter_month >= 1 else current_date.year - 1
    last_quarter = date(last_quarter_year, last_quarter_month, 1)


    #current_quarter = (current_date.month - 1) // 3 + 1
    #last_quarter_month = (current_quarter - 2) * 3
    #last_quarter_year = current_date.year if last_quarter_month >= 1 else current_date.year - 1
    #last_quarter = date(last_quarter_year, last_quarter_month, 1)

    # Calculate ISO 8601 date for this quarter
    this_quarter_month = (current_quarter - 1) * 3
    this_quarter = date(current_date.year, this_quarter_month, 1)

    # Calculate ISO 8601 date for next quarter
    next_quarter_month = (current_quarter + 1) * 3 if current_quarter != 4 else 1
    next_quarter_year = current_date.year if next_quarter_month <= 12 else current_date.year + 1
    next_quarter = date(next_quarter_year, next_quarter_month, 1)

    # Calculate ISO 8601 date for last semester
    last_semester_month = 1 if current_date.month <= 6 else 7
    last_semester_year = current_date.year if last_semester_month == 7 else current_date.year - 1
    last_semester = date(last_semester_year, last_semester_month, 1)
    """ 
    
    return {
        'yesterday': yesterday.isoformat(),
        'today': today.isoformat(),
        'tomorrow': tomorrow.isoformat(),
        #'last_night': last_night.isoformat(),
        #'this_morning': this_morning.isoformat(),
        #'next_week': next_week.isoformat(),
        #'last_week': last_week.isoformat(),
        #'this_month': this_month.isoformat(),
        #'next_month': next_month.isoformat(),
        #'last_month': last_month.isoformat(),
        #'this_year': this_year.isoformat(),
        #'next_year': next_year.isoformat(),
        #'last_year': last_year.isoformat(),
        #'last_weekend': f'{last_saturday.isoformat()} - {last_sunday.isoformat()}',
        #'this_weekend': f'{this_saturday.isoformat()} - {this_sunday.isoformat()}',
        #'next_weekend': f'{next_saturday.isoformat()} - {next_sunday.isoformat()}',
        #'last_quarter': last_quarter.isoformat(),
        #'this_quarter': this_quarter.isoformat(),
        #'next_quarter': next_quarter.isoformat(),
        #'last_semester': last_semester.isoformat()
    }

# Example usage:
#current_date = date(2023, 6, 22)  # Replace with your desired current date
#result = get_time_period_dates(current_date)
#for term, iso_date in result.items():
#    print(f'{term}: {iso_date}')
