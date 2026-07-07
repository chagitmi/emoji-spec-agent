from agent.graph import build_graph

app = build_graph()

result = app.invoke({"user_input": "פרצוף עם ראש של רובוט מרובע, עיניים בצורת מרובעים כחולים זוהרים, ופה שהוא קו ישר של אורות LED ירוקים", "retry_count": 0})

print("\n--- תוצאה סופית ---")
print(result)