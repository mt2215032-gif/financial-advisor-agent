def get_user_financial_data(user_id="user123"):
    return {
        "income": 70000,
        "monthly_expenses": {
            "housing": 1500,
            "food": 600,
            "transport": 300,
            "entertainment": 200,
            "others": 150,
        },
        "financial_goals": [
            "save for a house",
            "retire at 60"
        ],
        "risk_tolerance": "moderate"
    }