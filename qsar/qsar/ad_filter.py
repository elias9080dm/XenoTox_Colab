def filter_by_ad(df):
    """
    Filtra compuestos activos dentro del dominio de aplicabilidad
    """
    return df[
        (df["Prediction"] == "Active") &
        (df["AD_Flag"] == "IN")
    ].copy()
