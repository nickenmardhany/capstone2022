import React, { useContext } from "react";
import { useEffect } from "react";
import { useState } from "react";
import Login from "./components/Login";
import Header from "./components/Header";
import { UserContext } from "./context/UserContext";
import Table from "./components/Table";


const App= () => {
  const [message , setMessage] = useState("");
  const [token] = useContext(UserContext);
  const getWelcomeMessage = async ()=> {
    const requestOptions = {
      method: "GET",
      headers: {
        "Content-Type" : "application/json",
      },
    };
    const response = await fetch("/api", requestOptions);
    const data = await response.json();

    // console.log(JSON.stringify(data.data));
    if (!response.ok){
      console.log("something messed up");
    } else {
      setMessage(data.data);
    }
  }; 

  useEffect(()=> {
    getWelcomeMessage();
  }, []);

  return (
    <>
    <Header title={'Sistem Indirect Mass Reporting'}/>
    <div className="columns">
      <div className="column"></div>
      <div className="column m-5 is-two-thirds">
        {
          !token ? (
            <div className="columns">
               <Login/>
            </div>
          ) :(
            <Table/>
          )
        }
      </div>
        
      <div className="column"></div>
    </div>
    </>
    // <div>
    //   <h1>{message}</h1>
    //   <Register />
    // </div>
  );
};

export default App;
