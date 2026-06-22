import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || '/api'

const client = axios.create({ baseURL: BASE })

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('admin_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export const classify = (text) =>
  client.post('/classify', { text }).then((r) => r.data)

export const login = (username, password) =>
  client.post('/auth/login', { username, password }).then((r) => r.data)

export const getHistory = (page = 1, perPage = 20) =>
  client.get(`/history?page=${page}&per_page=${perPage}`).then((r) => r.data)

export const getStats = () =>
  client.get('/stats').then((r) => r.data)
