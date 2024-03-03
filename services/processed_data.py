from flask import jsonify
from . import supabase_api


def getProcessData():
    supabase = supabase_api.supabase
    try:
        first_test = supabase.table('first-test').select('*').execute().data
        second_test = supabase.table('second-test').select('*').execute().data
        third_test = supabase.table('third-test').select('*').execute().data
        fourth_test = supabase.table('fourth-test').select('*').execute().data
        fifth_test = supabase.table('fifth-test').select('*').execute().data

        status = 200
        message = 'Data fetch successfully'
        response = {
            'first': first_test, 
            'second': second_test, 
            'third': third_test, 
            'fourth': fourth_test, 
            'fifth' : fifth_test
        }

    except Exception as e: 
        status = 500
        message = 'There was an issue trying to fetch the data :('
        response = str(e)
    
    return jsonify({
        'status': status, 
        'message': message, 
        'response': response
    })
