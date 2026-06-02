export default {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
      "Access-Control-Max-Age": "86400",
    };

    // Handle CORS preflight requests
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders,
      });
    }

    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/api/scan") {
      try {
        // Parse the JSON request payload
        let body;
        try {
          body = await request.json();
        } catch (err) {
          return new Response(JSON.stringify({ error: "Invalid JSON syntax." }), {
            status: 400,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders,
            },
          });
        }

        // Forward request to Hugging Face space
        const apiResponse = await fetch("https://hassan2007-flood-intelligence-engine.hf.space/api/v1/analyze_flood", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${env.HF_TOKEN || ""}`
          },
          body: JSON.stringify(body),
        });

        const responseText = await apiResponse.text();
        let responseData;
        try {
          responseData = JSON.parse(responseText);
        } catch (err) {
          responseData = { error: responseText || "Upstream returned non-JSON content." };
        }

        return new Response(JSON.stringify(responseData), {
          status: apiResponse.status,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        });

      } catch (error) {
        return new Response(
          JSON.stringify({ error: "Failed to process satellite data via AI Engine." }),
          {
            status: 500,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders,
            },
          }
        );
      }
    } else if (request.method === "GET" && url.pathname.startsWith("/api/status/")) {
      try {
        const taskId = url.pathname.substring("/api/status/".length);
        if (!taskId) {
          return new Response(JSON.stringify({ error: "Task ID is required." }), {
            status: 400,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders,
            },
          });
        }

        // Forward GET request to Hugging Face space task status endpoint
        const apiResponse = await fetch(`https://hassan2007-flood-intelligence-engine.hf.space/api/v1/task_status/${taskId}`, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${env.HF_TOKEN || ""}`
          }
        });

        const responseText = await apiResponse.text();
        let responseData;
        try {
          responseData = JSON.parse(responseText);
        } catch (err) {
          responseData = { error: responseText || "Upstream returned non-JSON content." };
        }

        return new Response(JSON.stringify(responseData), {
          status: apiResponse.status,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders,
          },
        });

      } catch (error) {
        return new Response(
          JSON.stringify({ error: "Failed to retrieve task status from AI Engine." }),
          {
            status: 500,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders,
            },
          }
        );
      }
    } else {
      return new Response(JSON.stringify({ error: "Not Found" }), {
        status: 404,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders,
        },
      });
    }
  },
};
