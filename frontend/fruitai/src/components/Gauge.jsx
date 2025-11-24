import React from 'react'
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'


export default function Gauge({value=0}){
const color = value>70? '#4ade80': value>40 ? '#f59e0b' : '#fb7185'
return (
<div style={{width:220,height:220,display:'flex',alignItems:'center',justifyContent:'center'}}>
<CircularProgressbar value={value} text={`${Math.round(value)} %`} styles={buildStyles({ textColor:'#fff', pathColor: color, trailColor:'#263244'})} />
</div>
)
}