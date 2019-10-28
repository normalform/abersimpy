from material.get_attexp import get_attexp


def get_epsb(material, temp):
    epsb = get_attexp(material, temp);

    return epsb