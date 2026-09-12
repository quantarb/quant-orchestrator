"""Frozen annual option baskets and mechanical split adjustments."""
import polars as pl


def first_session_baskets(chain, first_session):
    first=chain.filter(pl.col('snapshot_date')==first_session).with_columns(
        (pl.col('expiration')-pl.lit(first_session)).dt.total_days().alias('dte'))
    first=first.filter(pl.col('dte')>0).unique('contract_symbol')
    selected=[]
    for right in ('call','put'):
        rows=first.filter(pl.col('option_type')==right)
        expiries=sorted(rows['dte'].unique().to_list())
        if len(expiries)<5:
            raise ValueError(f'First-session {right} cohort has {len(expiries)} DTEs; five required')
        chosen=[expiries[round(i*(len(expiries)-1)/4)] for i in range(5)]
        selected.append(rows.filter(pl.col('dte').is_in(chosen)))
    return (pl.concat(selected).with_columns(pl.concat_str([pl.lit('OPT_'),pl.col('underlying_symbol'),
        pl.lit(f'_{first_session.year}_'),pl.col('option_type').str.to_uppercase(),pl.lit('_DTE_'),pl.col('dte')]).alias('document_symbol'))
        .with_columns((1./pl.len().over('document_symbol')).alias('weight')))


def basket_quotes(quotes,members):
    joined=quotes.join(members.select('contract_symbol','document_symbol','weight'),on='contract_symbol',how='inner')
    joined=joined.filter((pl.col('bid')>=0)&(pl.col('ask')>0)&(pl.col('ask')>=pl.col('bid'))
        &pl.col('bid').is_finite()&pl.col('ask').is_finite()&(pl.col('snapshot_date')<=pl.col('expiration')))
    expected=members.group_by('document_symbol').len().rename({'len':'expected'})
    return (joined.unique(['document_symbol','contract_symbol','snapshot_date']).group_by('document_symbol','snapshot_date').agg(
        (pl.col('bid')*pl.col('weight')).sum().alias('low'),(pl.col('ask')*pl.col('weight')).sum().alias('high'),
        pl.col('volume').sum().alias('volume'),pl.len().alias('observed'))
        .join(expected,on='document_symbol').filter(pl.col('observed')==pl.col('expected'))
        .with_columns(((pl.col('low')+pl.col('high'))/2).alias('close'))
        .with_columns(pl.col('close').alias('open')).rename({'document_symbol':'symbol','snapshot_date':'date'}).sort('symbol','date'))


def split_adjusted_members(members, quotes, factor):
    if factor == 1:
        return members
    if factor <= 0:
        raise ValueError('Split ratio must be positive')
    # Integer forward splits multiply contract count and divide strike.
    # Match the exchange's rounded strike, rather than constructing an OCC
    # symbol with a guessed rounding convention (notably Apple's 7:1 split).
    original=members.rename({'contract_symbol':'original_contract_symbol','strike':'original_strike'}).with_columns(
        (pl.col('original_strike')/factor).alias('strike'),
        pl.col('original_contract_symbol').str.replace(r'\d{6}[CP]\d{8}$','').alias('_root'),
        (pl.col('weight')*factor).alias('weight'))
    available=quotes.select('expiration','option_type','strike','contract_symbol').unique().with_columns(
        pl.col('contract_symbol').str.replace(r'\d{6}[CP]\d{8}$','').alias('_root'),
        pl.col('strike').cast(pl.Float64)).sort('strike')
    original=original.with_columns(pl.col('expiration').cast(available.schema['expiration']),pl.col('strike').cast(pl.Float64))
    mapped=original.sort('strike').join_asof(available,on='strike',by=['expiration','option_type','_root'],
        strategy='nearest',tolerance=.011,check_sortedness=False)
    # Unmatched constituents retain an impossible identifier. Consequently the
    # complete-basket gate rejects their dates; it never reallocates weights.
    return mapped.with_columns(pl.col('contract_symbol').fill_null(pl.lit('MISSING_SPLIT_')+pl.col('original_contract_symbol'))).drop('_root')
