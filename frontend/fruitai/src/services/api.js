import axios from 'axios'
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'


export async function predictFruit(file){
const form = new FormData(); form.append('file', file)
const res = await axios.post(`${API_BASE}/predict/fruit`, form, { headers:{ 'Content-Type':'multipart/form-data' }})
return res.data
}
export async function predictRipeness(file, fruit){
const form = new FormData(); form.append('file', file); form.append('fruit', fruit||'')
const res = await axios.post(`${API_BASE}/predict/ripeness`, form, { headers:{ 'Content-Type':'multipart/form-data' }})
return res.data
}
export async function predictDisease(file, fruit){
const form = new FormData(); form.append('file', file); form.append('fruit', fruit||'')
const res = await axios.post(`${API_BASE}/predict/disease`, form, { headers:{ 'Content-Type':'multipart/form-data' }})
return res.data
}

export async function predictYOLO(file, conf=0.25){
	const form = new FormData(); form.append('file', file); form.append('conf', String(conf))
	const res = await axios.post(`${API_BASE}/predict/yolo`, form, { headers:{ 'Content-Type':'multipart/form-data' }})
	return res.data
}