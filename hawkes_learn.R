library(ptproc)

get_fitted_params = function(data) {
  ppm = ptproc(data, cond.int = hawkes.cond.int, params = c(mu = 0.5, C = 1.0, a = 0.1))
  condition(ppm) = penalty(code = NULL, condition = quote(any(params < 0)))
  fitted = ptproc.fit(ppm, alpha=1e+5)
  
  return (fitted$params)
}

estimate_Nt = function(params, l0, t){
  a = params[['a']]
  b = params[['C']]
  mu = params[['mu']]
  
  estimate_Nt = (l0 / (a - b) + b * mu / (a - b) ** 2) * exp((a - b) * t) - (b * mu / (a - b)) * t - (l0 / (a - b) + b * mu / (a - b) ** 2)
  
  A = (mu * (a ** 2 + 2 * b * mu) - 2 * mu * a * (a - b)) / (2 * (a - b) ** 2)
  B = (mu * (a ** 2 + 2 * b * mu) + 2 * mu * a * (a - b)) / (2 * (a - b) ** 2)
  C = (mu * b * mu) / (a - b) ** 2
  estimate_Nt2 = B / (2 * (a - b)) * exp(2 * (a - b) * t) - 2 * (A + B - C) / (a - b) * exp((a - b) * t)
  - C * (a - b) * t ** 2 + (mu + 2 * A + 2 * C) * t
  + (4 * A + 3 * B - 4 * C) / (2 * (a - b))
  
  return (estimate_Nt)
}

absolute_value_error = function(estimation, true) {
  return (abs(estimation - true))
}

diff_error = function(estimation, true){
    return (estimation - true)
}

CV = function(data, end_t, h=1, k=5) {
  first = data[1]
  last = data[length(data)]
  error = 0
  for (i in 1:k) {
    train_x = data[data <= (last - k + i - h)]
    y = length(data[(data > (last - k + i - h)) & (data <= (last - k + i))])
    Nt = forecast(train_x, end_t, h)
    
    #error = error + absolute_value_error(Nt, y)
    error = error + diff_error(Nt, y)
  }
  
  return (error / k)
}

forecast = function(train_data, end_t, h=1){
  params = get_fitted_params(train_data)
  l0 = params[['mu']]
  Nt = estimate_Nt(params, l0, h) - estimate_Nt(params, l0, h-1)

  return (data.frame(forecast=Nt, mu=params[['mu']], a=params[['a']], b=params[['C']]))
}