def feature_physics_mapping(feature_names):
    physics_map = {}

    for f in feature_names:
        if "amp" in f.lower():
            physics_map[f] = "Damage severity"
        elif "energy" in f.lower():
            physics_map[f] = "Energy release"
        elif "duration" in f.lower():
            physics_map[f] = "Crack growth time"
        elif "rise" in f.lower():
            physics_map[f] = "Crack initiation speed"
        elif "freq" in f.lower():
            physics_map[f] = "Damage mode"
        else:
            physics_map[f] = "Statistical descriptor"

    return physics_map

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("Mendeley_ae.csv")
    mapping = feature_physics_mapping(df.columns)
    for k, v in list(mapping.items())[:10]:
        print(k, "->", v)
