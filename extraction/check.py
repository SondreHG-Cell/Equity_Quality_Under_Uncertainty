import pandas as pd

# Try LSEG Data Library first
try:
    import lseg.data as ld
    from lseg.data.content import fundamental_and_reference as fr

    print("Using lseg.data")

    ld.open_session()

    test_ric = "AAK.ST"

    fields = [
        "TR.DivAdjustmentFactor"
    ]

    rows = []

    for field in fields:
        try:
            response = fr.Definition(
                universe=[test_ric],
                fields=[field],
                parameters={
                    "SDate": "2020-01-01",
                    "EDate": "2020-12-31",
                    "Frq": "M",
                },
            ).get_data()

            df = response.data.df

            rows.append({
                "Field": field,
                "Accessible": True,
                "Rows": len(df),
                "NonMissingValues": df[field].notna().sum() if field in df.columns else None,
                "ColumnsReturned": list(df.columns),
                "ExampleValue": df[field].dropna().iloc[0] if field in df.columns and df[field].notna().any() else None,
                "Error": None,
            })

        except Exception as e:
            rows.append({
                "Field": field,
                "Accessible": False,
                "Rows": None,
                "NonMissingValues": None,
                "ColumnsReturned": None,
                "ExampleValue": None,
                "Error": str(e)[:300],
            })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    result.to_csv("lseg_field_access_check.csv", index=False)
    print("\nSaved to lseg_field_access_check.csv")

    ld.close_session()

except Exception as e:
    print("lseg.data test failed:")
    print(e)